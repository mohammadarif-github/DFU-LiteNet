package com.dfulitenet.demo;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ImageDecoder;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * DFU-LiteNet demo: classifies a skin image as ulcer or healthy with the thesis model
 * (MobileNetV3-Small, FP16 TensorFlow Lite, fold-0 model) and shows the Grad-CAM heatmap that the
 * model outputs together with its prediction. Everything runs on the phone; no internet is used.
 */
public class MainActivity extends Activity {

    private static final String MODEL_FILE = "dfu_litenet_fp16.tflite";
    private static final float THRESHOLD = 0.49f;   // chosen on the validation fold (notebook 09, fold 0)
    private static final int SIZE = 224;            // model input: 224 x 224 RGB, raw 0-255 values
    private static final int DISPLAY = 672;         // size of the picture shown on screen
    private static final int TIMED_RUNS = 5;        // the reported time is the median of these runs
    private static final int REQ_GALLERY = 1, REQ_CAMERA = 2;

    private static final String[][] SAMPLES = {
            {"samples/ulcer_1.jpg", "Test sample: ulcer"},
            {"samples/healthy_1.jpg", "Test sample: healthy skin"},
            {"samples/ulcer_2.jpg", "Test sample: ulcer"},
            {"samples/healthy_2.jpg", "Test sample: healthy skin"},
    };

    private Interpreter tflite;
    private int probIndex = 0, heatIndex = 1;
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler ui = new Handler(Looper.getMainLooper());

    private ImageView imageView;
    private TextView placeholder, resultLabel, resultProb, resultTime;
    private ProgressBar probBar;
    private View resultCard;
    private Switch heatmapSwitch;
    private Bitmap shownPlain, shownWithHeat;
    private Uri cameraUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        placeholder = findViewById(R.id.placeholder);
        resultCard = findViewById(R.id.resultCard);
        resultLabel = findViewById(R.id.resultLabel);
        resultProb = findViewById(R.id.resultProb);
        resultTime = findViewById(R.id.resultTime);
        probBar = findViewById(R.id.probBar);
        heatmapSwitch = findViewById(R.id.heatmapSwitch);

        ((TextView) findViewById(R.id.aboutText)).setText(
                "Model: MobileNetV3-Small (0.94 M parameters), TensorFlow Lite FP16, 1.93 MB. Trained and tested "
                + "with a leakage-free 5-fold split; F1 = 99.4% over the 5 test folds. The heatmap is the model's own "
                + "Grad-CAM output, computed in the same run as the prediction.\n\n"
                + "Tip: the model was trained on close-up skin patches. For your own photos, take a close-up of the "
                + "skin area, like the samples.\n\n"
                + "Research demo for the CSE-4800 thesis at IIUC. Not a medical diagnosis.");

        heatmapSwitch.setOnCheckedChangeListener((b, checked) -> showPicture());
        ((Button) findViewById(R.id.galleryButton)).setOnClickListener(v -> openGallery());
        ((Button) findViewById(R.id.cameraButton)).setOnClickListener(v -> openCamera());
        buildSamples();

        worker.execute(() -> {
            try {
                loadModel();
            } catch (Exception e) {
                ui.post(() -> toast("Could not load the model: " + e.getMessage()));
            }
        });
    }

    // ------------------------------------------------------------------ model
    private void loadModel() throws Exception {
        MappedByteBuffer buffer;
        try (AssetFileDescriptor fd = getAssets().openFd(MODEL_FILE);
             FileInputStream in = new FileInputStream(fd.getFileDescriptor())) {
            buffer = in.getChannel().map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
        }
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        tflite = new Interpreter(buffer, options);
        // two outputs: the ulcer probability [1, 1] and the 7 x 7 heatmap [1, 7, 7, 1]
        for (int i = 0; i < tflite.getOutputTensorCount(); i++) {
            if (tflite.getOutputTensor(i).shape().length == 4) heatIndex = i; else probIndex = i;
        }
        run(ByteBuffer.allocateDirect(SIZE * SIZE * 3 * 4).order(ByteOrder.nativeOrder()));   // warm-up
    }

    private static final class Result {
        float probability;
        float[][] heat = new float[7][7];
        double millis;
    }

    private Result run(ByteBuffer input) {
        float[][] prob = new float[1][1];
        float[][][][] heat = new float[1][7][7][1];
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(probIndex, prob);
        outputs.put(heatIndex, heat);
        input.rewind();
        long t0 = System.nanoTime();
        tflite.runForMultipleInputsOutputs(new Object[]{input}, outputs);
        Result r = new Result();
        r.millis = (System.nanoTime() - t0) / 1e6;
        r.probability = prob[0][0];
        for (int y = 0; y < 7; y++) for (int x = 0; x < 7; x++) r.heat[y][x] = heat[0][y][x][0];
        return r;
    }

    private void analyze(Bitmap source, String sourceName) {
        if (tflite == null) {
            toast("The model is still loading, try again in a second");
            return;
        }
        worker.execute(() -> {
            Bitmap square = centerSquare(source);
            Bitmap input224 = Bitmap.createScaledBitmap(square, SIZE, SIZE, true);
            ByteBuffer input = toInput(input224);

            double[] times = new double[TIMED_RUNS];
            Result result = null;
            for (int i = 0; i < TIMED_RUNS; i++) {
                result = run(input);
                times[i] = result.millis;
            }
            Arrays.sort(times);
            double median = times[TIMED_RUNS / 2];

            Bitmap plain = Bitmap.createScaledBitmap(square, DISPLAY, DISPLAY, true);
            Bitmap withHeat = overlay(plain, result.heat);
            final Result r = result;
            ui.post(() -> {
                shownPlain = plain;
                shownWithHeat = withHeat;
                showPicture();
                showResult(r, median, sourceName);
            });
        });
    }

    /** Model input: 224 x 224 x 3 floats with raw 0-255 pixel values (the model does its own scaling). */
    private static ByteBuffer toInput(Bitmap bmp) {
        int[] px = new int[SIZE * SIZE];
        bmp.getPixels(px, 0, SIZE, 0, 0, SIZE, SIZE);
        ByteBuffer buf = ByteBuffer.allocateDirect(SIZE * SIZE * 3 * 4).order(ByteOrder.nativeOrder());
        for (int p : px) {
            buf.putFloat((p >> 16) & 0xFF);
            buf.putFloat((p >> 8) & 0xFF);
            buf.putFloat(p & 0xFF);
        }
        return buf;
    }

    private static Bitmap centerSquare(Bitmap src) {
        int side = Math.min(src.getWidth(), src.getHeight());
        int x = (src.getWidth() - side) / 2, y = (src.getHeight() - side) / 2;
        return Bitmap.createBitmap(src, x, y, side, side);
    }

    /** Colour the 7 x 7 heatmap (jet colours, stronger = more opaque), smooth it to the picture size, draw it on top. */
    private static Bitmap overlay(Bitmap base, float[][] heat) {
        float max = 0f;
        for (float[] row : heat) for (float v : row) max = Math.max(max, v);
        Bitmap out = base.copy(Bitmap.Config.ARGB_8888, true);
        if (max <= 0f) return out;
        Bitmap small = Bitmap.createBitmap(7, 7, Bitmap.Config.ARGB_8888);
        for (int y = 0; y < 7; y++) {
            for (int x = 0; x < 7; x++) {
                float v = heat[y][x] / max;
                int r = (int) (255 * clamp(1.5f - Math.abs(4 * v - 3)));
                int g = (int) (255 * clamp(1.5f - Math.abs(4 * v - 2)));
                int b = (int) (255 * clamp(1.5f - Math.abs(4 * v - 1)));
                int a = (int) (60 + 130 * v);
                small.setPixel(x, y, (a << 24) | (r << 16) | (g << 8) | b);
            }
        }
        Bitmap big = Bitmap.createScaledBitmap(small, base.getWidth(), base.getHeight(), true);
        new Canvas(out).drawBitmap(big, 0, 0, new Paint(Paint.FILTER_BITMAP_FLAG));
        return out;
    }

    private static float clamp(float v) {
        return Math.max(0f, Math.min(1f, v));
    }

    // ------------------------------------------------------------------ UI
    private void showPicture() {
        if (shownPlain == null) return;
        placeholder.setVisibility(View.GONE);
        imageView.setImageBitmap(heatmapSwitch.isChecked() ? shownWithHeat : shownPlain);
    }

    private void showResult(Result r, double millis, String sourceName) {
        boolean ulcer = r.probability >= THRESHOLD;
        int colour = getColor(ulcer ? R.color.orange : R.color.green);
        resultCard.setVisibility(View.VISIBLE);
        resultLabel.setText(ulcer ? "Ulcer detected" : "Healthy skin");
        resultLabel.setTextColor(colour);
        resultProb.setText(String.format(Locale.US, "Ulcer probability: %.1f%%  (decision threshold %.0f%%)",
                100 * r.probability, 100 * THRESHOLD));
        probBar.setProgress(Math.round(r.probability * 1000));
        probBar.setProgressTintList(ColorStateList.valueOf(colour));
        float heatMax = 0f;
        for (float[] row : r.heat) for (float v : row) heatMax = Math.max(heatMax, v);
        String heatNote = heatMax > 0f
                ? "Heatmap: red = the area that made the model say \"ulcer\"."
                : "Heatmap: no ulcer evidence found, so nothing is highlighted.";
        resultTime.setText(String.format(Locale.US,
                "%s\nInference: %.1f ms on this phone (CPU, 4 threads, median of %d runs).\n%s",
                sourceName, millis, TIMED_RUNS, heatNote));
    }

    private void buildSamples() {
        LinearLayout row = findViewById(R.id.samplesRow);
        int size = (int) (76 * getResources().getDisplayMetrics().density);
        for (int i = 0; i < SAMPLES.length; i++) {
            final String file = SAMPLES[i][0], name = SAMPLES[i][1];
            try (InputStream in = getAssets().open(file)) {
                Bitmap bmp = BitmapFactory.decodeStream(in);
                ImageView iv = new ImageView(this);
                LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, size, 1f);
                if (i > 0) lp.setMarginStart((int) (8 * getResources().getDisplayMetrics().density));
                iv.setLayoutParams(lp);
                iv.setScaleType(ImageView.ScaleType.CENTER_CROP);
                iv.setImageBitmap(bmp);
                iv.setContentDescription(name);
                iv.setOnClickListener(v -> analyze(bmp, name));
                row.addView(iv);
            } catch (Exception e) {
                toast("Missing sample " + file);
            }
        }
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(Intent.createChooser(intent, "Choose a skin photo"), REQ_GALLERY);
    }

    private void openCamera() {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, "dfu_" + System.currentTimeMillis() + ".jpg");
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/DFU-LiteNet");
        cameraUri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        intent.putExtra(MediaStore.EXTRA_OUTPUT, cameraUri);
        intent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        try {
            startActivityForResult(intent, REQ_CAMERA);
        } catch (Exception e) {
            toast("No camera app found");
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        Uri uri = null;
        if (requestCode == REQ_GALLERY && resultCode == RESULT_OK && data != null) uri = data.getData();
        if (requestCode == REQ_CAMERA) {
            if (resultCode == RESULT_OK) {
                uri = cameraUri;
            } else if (cameraUri != null) {
                getContentResolver().delete(cameraUri, null, null);   // photo cancelled: remove the empty entry
            }
        }
        if (uri == null) return;
        final Uri chosen = uri;
        final String name = requestCode == REQ_CAMERA ? "Your camera photo" : "Your gallery photo";
        worker.execute(() -> {
            try {
                Bitmap bmp = decode(chosen);
                ui.post(() -> analyze(bmp, name));
            } catch (Exception e) {
                ui.post(() -> toast("Could not open the image"));
            }
        });
    }

    /** Decodes a photo, applying its rotation, and keeps it at most about 1600 pixels on the long side. */
    private Bitmap decode(Uri uri) throws Exception {
        ImageDecoder.Source src = ImageDecoder.createSource(getContentResolver(), uri);
        return ImageDecoder.decodeBitmap(src, (decoder, info, s) -> {
            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
            int w = info.getSize().getWidth(), h = info.getSize().getHeight();
            float scale = Math.min(1f, 1600f / Math.max(w, h));
            if (scale < 1f) decoder.setTargetSize(Math.round(w * scale), Math.round(h * scale));
        });
    }

    private void toast(String msg) {
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        worker.shutdown();
        if (tflite != null) tflite.close();
    }
}
