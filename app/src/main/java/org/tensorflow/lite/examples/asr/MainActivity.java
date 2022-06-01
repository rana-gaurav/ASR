package org.tensorflow.lite.examples.asr;

import static android.Manifest.permission.MANAGE_EXTERNAL_STORAGE;
import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.RECORD_AUDIO;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;

import android.Manifest;
import android.content.ContextWrapper;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.MediaPlayer;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.LongDef;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.jlibrosa.audio.JLibrosa;
import com.jlibrosa.audio.exception.FileFormatNotSupportedException;
import com.jlibrosa.audio.wavFile.WavFileException;

import org.jtransforms.fft.FloatFFT_1D;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;

public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    private final static String TAG = "MainActivity";

    private Spinner audioClipSpinner;
    private Button transcribeButton;
    private Button playAudioButton;
    private Button stopAudioButton;
    private Button playCleanButton;
    private ProgressBar progressBar;
    private TextView resultTextview;
    private ScrollView scrollView;


    private int numBlocks;
    private int blockShift = 128;
    private int blockLength = 512;
    private int count = 0;
    private final static int SAMPLE_RATE = 16000;
    private final static int DEFAULT_AUDIO_DURATION = -1;
    protected static final int BYTES_PER_FLOAT = Float.SIZE / 8;

    private final static String[] WAV_FILENAMES = {"audio_cut.wav", "ajay.wav", "a1.wav", "a2.wav", "door1.wav", "door2.wav", "door3.wav", "door4.wav", "door5.wav", "door6.wav", "door7.wav", "door8.wav", "door9.wav"};
    private final static String TFLITE_FILE_1 = "model_1.tflite";
    private final static String TFLITE_FILE_2 = "model_2.tflite";
    private String wavFilename;

    float[] audioFeatureValues;
    float[][] chunkData;
    float[][] inBuffer;
    float[][][] inputShape1;
    float[][][][] inputShape2;

    // Output of tflite 1 and should be initialized to f2;
    float[][][][] hashMapOutputB;
    float[][][][] hashMapOutputD;

    // final outputs from models
    float[] outputOfModel1, outputOfModel2;
    float[] outputBuffer = new float[512];
    float[] completeBuffer;

    Map<Integer, Object> outputMap1, outputMap2;

    private MappedByteBuffer tfLiteModel1, tfLiteModel2;
    private Interpreter tfLite1, tfLite2;
    private MediaPlayer mediaPlayer, mp;
    JLibrosa jLibrosa;

    File saveDir = new File(Environment.getExternalStorageDirectory().getAbsolutePath() + "/ActiveNoise");
    File originalDir = new File(saveDir.getAbsolutePath() + "/original");
    File cleanDir = new File(saveDir.getAbsolutePath() + "/clean");
    File cleanAudioFIle = new File(cleanDir.getAbsolutePath() + "/cleanAudio.wav");


    private double total = 0;
    private double avg = 0;
    private StringBuilder stringBuilder=new StringBuilder();



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (!checkPermission()) {
            requestPermission();
        }

        jLibrosa = new JLibrosa();
        initViews();
        playAudioButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                try (AssetFileDescriptor assetFileDescriptor = getAssets().openFd(wavFilename)) {
                    mediaPlayer.reset();
                    mediaPlayer.setDataSource(assetFileDescriptor.getFileDescriptor(), assetFileDescriptor.getStartOffset(), assetFileDescriptor.getLength());
                    mediaPlayer.prepare();
                } catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }
                mediaPlayer.start();
            }
        });

        transcribeButton.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {

                AsyncTaskExample asyncTask = new AsyncTaskExample();
                asyncTask.execute();

            }
        });

        playCleanButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    if (mediaPlayer.isPlaying()) {
                        mediaPlayer.stop();
                    }
                    //mp.reset();
                    mp.setDataSource(getFilePath());
                    mp.prepare();
                } catch (IllegalArgumentException e) {
                    e.printStackTrace();
                } catch (Exception e) {
                    System.out.println("Exception of type : " + e.toString());
                    e.printStackTrace();
                }
                mp.start();
            }
        });

        stopAudioButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (mediaPlayer != null || mp != null) {
                    mediaPlayer.pause();
                    mp.pause();
                    // mediaPlayer.release();
                }
            }
        });

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case 1:
                if (grantResults.length > 0) {
                    boolean StoragePermission = grantResults[0] ==
                            PackageManager.PERMISSION_GRANTED;
                    boolean RecordPermission = grantResults[1] ==
                            PackageManager.PERMISSION_GRANTED;

                    if (StoragePermission && RecordPermission) {
                        Toast.makeText(MainActivity.this, "Permission Granted",
                                Toast.LENGTH_SHORT).show();
                    } else {
                        //Toast.makeText(MainActivity.this, "Enable External file permissions", Toast.LENGTH_SHORT).show();
                    }
                }
                break;
        }
    }

    private void initViews() {
        ArrayAdapter<String> adapter = new ArrayAdapter<>(MainActivity.this,
                android.R.layout.simple_spinner_item, WAV_FILENAMES);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        audioClipSpinner = findViewById(R.id.audio_clip_spinner);
        audioClipSpinner.setAdapter(adapter);
        audioClipSpinner.setOnItemSelectedListener(this);
        playAudioButton = findViewById(R.id.play);
        transcribeButton = findViewById(R.id.recognize);
        resultTextview = findViewById(R.id.result);

        stopAudioButton = findViewById(R.id.btnStop);
        playCleanButton = findViewById(R.id.btnPlayClean);
        progressBar = findViewById(R.id.progressBar);
        scrollView = findViewById(R.id.sc_view);



        mediaPlayer = new MediaPlayer();
        mp = new MediaPlayer();

        if (!saveDir.exists()) saveDir.mkdirs();
        if (!originalDir.exists()) originalDir.mkdirs();
        if (!cleanDir.exists()) cleanDir.mkdirs();
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View v, int position, long id) {
        wavFilename = WAV_FILENAMES[position];
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {
    }

    @Override
    protected void onStop() {
        super.onStop();

        tfLite1.close();
        tfLite2.close();
    }

    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename) throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private String copyWavFileToCache(String wavFilename) {
        File destinationFile = new File(getCacheDir() + wavFilename);
        if (!destinationFile.exists()) {
            try {
                InputStream inputStream = getAssets().open(wavFilename);
                int inputStreamSize = inputStream.available();
                byte[] buffer = new byte[inputStreamSize];
                inputStream.read(buffer);
                inputStream.close();
                FileOutputStream fileOutputStream = new FileOutputStream(destinationFile);
                fileOutputStream.write(buffer);
                fileOutputStream.close();
            } catch (Exception e) {
                Log.e(TAG, e.getMessage());
            }
        }
        return getCacheDir() + wavFilename;
    }

    public static float[][] ArrayChunk(float[] array, int chunkSize) {
        int numOfChunks = (int) Math.ceil((double) array.length / chunkSize);
        float[][] output = new float[numOfChunks][];
        for (int i = 0; i < numOfChunks; i++) {
            int start = i * chunkSize;
            int length = Math.min(array.length - start, chunkSize);

            float[] temp = new float[length];
            System.arraycopy(array, start, temp, 0, length);
            output[i] = temp;
        }
        return output;
    }

    private float[][][] inputShapeA(float[][] input) {
        inputShape1 = new float[1][1][257];
        for (int i = 0; i < input[0].length; i++) {
            inputShape1[0][0][i] = (input[0][i]);
        }
        if (input[0].length == 256) {
            inputShape1[0][0][256] = input[0][255];
        }
        return inputShape1;
    }

    // shape for model 2
    private float[][][] inputShapeC(float[][] input) {
        inputShape1 = new float[1][1][512];
        for (int i = 0; i < input[0].length; i++) {
            inputShape1[0][0][i] = (input[0][i]);
        }
        /*if (input[0].length == 512) {
            inputShape1[0][0][256] = input[0][255];
        }*/
        return inputShape1;
    }

    private float[][][][] inputShapeB(String input) {
        if (input == null) {
            inputShape2 = new float[][][][]{{{{0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}},
                    {{0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f}}}};
        } else {
            inputShape2 = hashMapOutputB;
        }
        return inputShape2;
    }

    private float[][][][] inputShapeD(String input) {
        if (input == null) {
            inputShape2 = new float[][][][]{{{{0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}},
                    {{0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f}}}};
        } else {
            inputShape2 = hashMapOutputD;
        }
        return inputShape2;
    }

    public void initOutput1() {
        // IntBuffer outputBuffer = IntBuffer.allocate(2000);
        outputMap1 = new HashMap<>();
        float[][][] out1 = new float[1][1][257];
        float[][][][] out2 = new float[1][2][128][2];
        outputMap1.put(0, out1);
        outputMap1.put(1, out2);
    }

    public void initOutput2() {
        // IntBuffer outputBuffer = IntBuffer.allocate(2000);
        outputMap2 = new HashMap<>();
        float[][][] out1 = new float[1][1][512];
        float[][][][] out2 = new float[1][2][128][2];
        outputMap2.put(0, out1);
        outputMap2.put(1, out2);
    }

    public float[] getPhaseAngle(ArrayList<Float> real, ArrayList<Float> img) {
        float[] phaseAngle = new float[real.size()];
        for (int i = 0; i < real.size(); i++) {
            phaseAngle[i] = (float) Math.atan2(img.get(i), real.get(i));
        }
        return phaseAngle;
    }

    private float[] getAbs(ArrayList<Float> real, ArrayList<Float> img) {
        float[] abs = new float[real.size()];
        for (int i = 0; i < real.size(); i++) {
            abs[i] = (float) Math.sqrt(Math.pow(real.get(i), 2) + (Math.pow(img.get(i), 2)));
        }
        return abs;
    }

    private void initTflite1(String model) throws IOException {
        tfLiteModel1 = loadModelFile(getAssets(), model);
        Interpreter.Options tfLiteOptions1 = new Interpreter.Options();

        /*CompatibilityList compatList= new CompatibilityList();
        if(compatList.isDelegateSupportedOnThisDevice()){
            GpuDelegate.Options delegateOptions= compatList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate=new GpuDelegate(delegateOptions);
            tfLiteOptions1.addDelegate(gpuDelegate);
        }else {
            // if the GPU is not supported, it will run on 4 threads
            tfLiteOptions1.setNumThreads(4);
        }
*/
        tfLite1 = new Interpreter(tfLiteModel1, tfLiteOptions1);
    }

    private void feedTFLite1(float[][][] f1, float[][][][] f2) {
        Object[] inputArray = {f1, f2};
        tfLite1.runForMultipleInputsOutputs(inputArray, outputMap1);
        tfliteOutput1(outputMap1);
        Log.d("XXX", "Success");
    }

    private void initTflite2(String model) throws IOException {
        tfLiteModel2 = loadModelFile(getAssets(), model);
        Interpreter.Options tfLiteOptions2 = new Interpreter.Options();

       /* CompatibilityList compatList= new CompatibilityList();
        if(compatList.isDelegateSupportedOnThisDevice()){
            //GpuDelegate.Options delegateOptions= compatList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate=new GpuDelegate(delegateOptions);
            tfLiteOptions2.addDelegate(gpuDelegate);
            Log.d("XXX","Using GPU");
        }else {
            // if the GPU is not supported, it will run on 4 threads
            tfLiteOptions2.setNumThreads(4);
            Log.d("XXX","Not using GPU");
        }
*/
        tfLite2 = new Interpreter(tfLiteModel2, tfLiteOptions2);
    }

    private void feedTFLite2(float[][][] f1, float[][][][] f2, int index) {
        Object[] inputArray = {f1, f2};
        tfLite2.runForMultipleInputsOutputs(inputArray, outputMap2);
        tfliteOutput2(outputMap2, index);
        Log.d("XXX", "Success");
    }

    private ArrayList<Float> getPart(String part, float[] data) {
        ArrayList<Float> real = new ArrayList<>();
        ArrayList<Float> img = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            if (i % 2 == 0) {
                //Even
                real.add(data[i]);
            } else {
                //Odd
                img.add(data[i]);
            }
        }
        if (part.equals("real")) {
            return real;
        } else {
            return img;
        }
    }

    private float[] realForwardFT(float[] floats) {
        FloatFFT_1D floatFFT_1D = new FloatFFT_1D(floats.length);
        floatFFT_1D.realForward(floats);
        return floats;
    }

    private float[] realInverseFT(float[] data) {
        FloatFFT_1D floatFFT_1D = new FloatFFT_1D(data.length);
        floatFFT_1D.realInverse(data, true);
        return data;
    }

    private float[] estimatedComplex(float[] abs, float[] outputOfModel1, float[] phaseAngle) {

        float[] real = new float[abs.length];
        float[] img = new float[abs.length];
        float[] estimatedValues = new float[real.length + img.length];
        int j = 0;

        // cal the estimated values fro eluer methods
        for (int i = 0; i < abs.length; i++) {
            real[i] = (float) (abs[i] * outputOfModel1[i] * Math.cos((phaseAngle[i])));
            img[i] = (float) (abs[i] * outputOfModel1[i] * Math.sin((phaseAngle[i])));
        }

        // combine the real and img in a sequence of array
        for (int i = 0; i < real.length + img.length; i += 2) {
            estimatedValues[i] = real[j];
            estimatedValues[i + 1] = img[j];
            j += 1;
        }

        Log.d("estimated complex", "");
        return estimatedValues;
    }

    private void tfliteOutput1(Map<Integer, Object> outputMap) {
        float[][][] hashMapOutput1 = (float[][][]) outputMap.get(0);
        hashMapOutputB = (float[][][][]) outputMap.get(1);
        // outputModel1 is 1d array extract from 3d array used for estimated complex.
        outputOfModel1 = hashMapOutput1[0][0];
        Log.d("cc", "" + outputOfModel1);
    }

    private void tfliteOutput2(Map<Integer, Object> outputMap, int index) {
        float[][][] hashMapOutput = (float[][][]) outputMap.get(0);
        hashMapOutputD = (float[][][][]) outputMap.get(1);
        outputOfModel2 = hashMapOutput[0][0];
        try {
            getData(outputOfModel2, index);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Log.d("data1", "" + hashMapOutputD.length);
    }

    private void getData(float[] lastOutput, int index) throws IOException {
        Log.d("XXX", "" + index);
        float[] emptyBuffer = new float[128];
        System.arraycopy(outputBuffer, 128, outputBuffer, 0, 384);
        System.arraycopy(emptyBuffer, 0, outputBuffer, 384, emptyBuffer.length);
        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = outputBuffer[i] + lastOutput[i];
        }
        System.arraycopy(outputBuffer, 0, completeBuffer, (index * 128), 128);
        Log.d("XXX", String.valueOf(completeBuffer));
    }

    public void writeFloatToByte(float[] array) throws IOException {
        byte[] audioBuffer = new byte[array.length * 4];

        /*for (int i = 0; i < array.length; i++) {
            byte[] byteArray = ByteBuffer.allocate(4).putFloat(array[i] * 32767).array();
            for (int k = 0; k < byteArray.length; k++) {
                audioBuffer[i * 4 + k] = byteArray[k];
            }
        }*/

        /*for (int x = 0; x < array.length; x++) {
            ByteBuffer.wrap(audioBuffer, x*4, 4).putFloat(array[x]);
        }
*/

        ByteBuffer buffer = ByteBuffer.allocate(BYTES_PER_FLOAT * array.length).
                order(ByteOrder.LITTLE_ENDIAN);
        for (float f : array) {
            //buffer.putFloat(f);
            buffer.putShort((short) (f * 32768F));
        }
        audioBuffer = buffer.array();

        Log.d("XXX", Arrays.toString(audioBuffer));


        // convert audio buffer to wav buffer by header to it.
        byte[] mainAudioData = convertWaveFile(audioBuffer);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {

            // writing the final byte array to file to creating the wav file
            writeByteToFile(mainAudioData);

        } else {
            // Request permission from the user
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 0);
        }

        Log.d("XXX", Arrays.toString(mainAudioData));

    }

    void writeByteToFile(byte[] bytes) {
        // Try block to check for exceptions
        try {
            // Initialize a pointer in file
            // using OutputStream
            DataOutputStream os = new DataOutputStream(new FileOutputStream(getFilePath()));
            // Starting writing the bytes in it
            os.write(bytes);
            // Close the file connections
            os.close();

        }
        // Catch block to handle the exceptions
        catch (Exception e) {
            Log.d("time", e.getMessage());
            Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    public byte[] convertWaveFile(byte[] data) {
        FileOutputStream out = null;
        long totalAudioLen = 0;
        long totalDataLen = totalAudioLen + 36;
        long longSampleRate = 16000;
        int channels = 1;
        long byteRate = 16 * longSampleRate * channels / 8;
        try {
            totalAudioLen = data.length;
            totalDataLen = totalAudioLen + 36;
            byte[] header = writeWaveFileHeader(totalAudioLen, totalDataLen, longSampleRate, channels, byteRate);
            byte[] result = new byte[header.length + data.length];
            System.arraycopy(header, 0, result, 0, header.length);
            System.arraycopy(data, 0, result, header.length, data.length);
            return result;

        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    private byte[] writeWaveFileHeader(long totalAudioLen, long totalDataLen, long longSampleRate, int channels, long byteRate) throws IOException {
        byte[] header = new byte[44];
        header[0] = 'R'; // RIFF
        header[1] = 'I';
        header[2] = 'F';
        header[3] = 'F';
        header[4] = (byte) (totalDataLen & 0xff);//数据大小
        header[5] = (byte) ((totalDataLen >> 8) & 0xff);
        header[6] = (byte) ((totalDataLen >> 16) & 0xff);
        header[7] = (byte) ((totalDataLen >> 24) & 0xff);
        header[8] = 'W';//WAVE
        header[9] = 'A';
        header[10] = 'V';
        header[11] = 'E';
        //FMT Chunk
        header[12] = 'f'; // 'fmt '
        header[13] = 'm';
        header[14] = 't';
        header[15] = ' ';//过渡字节
        //数据大小
        header[16] = 16; // 4 bytes: size of 'fmt ' chunk
        header[17] = 0;
        header[18] = 0;
        header[19] = 0;
        //编码方式 10H为PCM编码格式
        header[20] = 1; // format = 1
        header[21] = 0;
        //通道数
        header[22] = (byte) channels;
        header[23] = 0;
        //采样率，每个通道的播放速度
        header[24] = (byte) (longSampleRate & 0xff);
        header[25] = (byte) ((longSampleRate >> 8) & 0xff);
        header[26] = (byte) ((longSampleRate >> 16) & 0xff);
        header[27] = (byte) ((longSampleRate >> 24) & 0xff);
        //音频数据传送速率,采样率*通道数*采样深度/8
        header[28] = (byte) (byteRate & 0xff);
        header[29] = (byte) ((byteRate >> 8) & 0xff);
        header[30] = (byte) ((byteRate >> 16) & 0xff);
        header[31] = (byte) ((byteRate >> 24) & 0xff);
        // 确定系统一次要处理多少个这样字节的数据，确定缓冲区，通道数*采样位数
        header[32] = (byte) (1 * 16 / 8);
        header[33] = 0;
        //每个样本的数据位数
        header[34] = 16;
        header[35] = 0;
        //Data chunk
        header[36] = 'd';//data
        header[37] = 'a';
        header[38] = 't';
        header[39] = 'a';
        header[40] = (byte) (totalAudioLen & 0xff);
        header[41] = (byte) ((totalAudioLen >> 8) & 0xff);
        header[42] = (byte) ((totalAudioLen >> 16) & 0xff);
        header[43] = (byte) ((totalAudioLen >> 24) & 0xff);
        return header;
    }

    public boolean checkPermission() {
        int result = ContextCompat.checkSelfPermission(getApplicationContext(),
                WRITE_EXTERNAL_STORAGE);
        int result1 = ContextCompat.checkSelfPermission(getApplicationContext(),
                MANAGE_EXTERNAL_STORAGE);
        return result == PackageManager.PERMISSION_GRANTED &&
                result1 == PackageManager.PERMISSION_GRANTED;
    }

    private void requestPermission() {
        ActivityCompat.requestPermissions(MainActivity.this, new
                String[]{WRITE_EXTERNAL_STORAGE, MANAGE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, RECORD_AUDIO}, 1);
    }

    private String getFilePath() {
        ContextWrapper cw = new ContextWrapper(getApplicationContext());
        File f = cw.getExternalFilesDir(Environment.DIRECTORY_MUSIC);
        File file = new File(f, "sample" + ".wav");

        return file.getPath();
    }


    private class AsyncTaskExample extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void... voids) {

            ArrayList<Double> pre1 = new ArrayList<>();
            ArrayList<Double> pre2 = new ArrayList<>();
            ArrayList<Double> model1time = new ArrayList<>();
            ArrayList<Double> model2time = new ArrayList<>();
            ArrayList<Double> audioRead = new ArrayList<>();
            ArrayList<Double> blockShiftTime = new ArrayList<>();
            ArrayList<Double> fft = new ArrayList<>();
            ArrayList<Double> absEStTime = new ArrayList<>();

            final long totalstart = System.currentTimeMillis();

            try {

                long readAudioFileStart = System.currentTimeMillis();

                ExecutorService executorService = Executors.newSingleThreadExecutor();
                executorService.submit(new Runnable() {
                    @Override
                    public void run() {
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                // full audio buffer
                                try {
                                    audioFeatureValues = jLibrosa.loadAndRead(copyWavFileToCache(wavFilename), SAMPLE_RATE, DEFAULT_AUDIO_DURATION);
                                } catch (IOException | WavFileException | FileFormatNotSupportedException e) {
                                    e.printStackTrace();
                                }


                            }
                        });
                    }
                });



                long readAudioFileEnd = System.currentTimeMillis();
                Log.d("time", "The process was running: full audio buffer:- " + ((double) (readAudioFileEnd - readAudioFileStart) / 1000.0d) + "sec");
                audioRead.add((double) (readAudioFileEnd - readAudioFileStart) / 1000.0d);

                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        resultTextview.append("The process was running: full audio buffer:- " + ((double) (readAudioFileEnd - readAudioFileStart) / 1000.0d) + "sec");
                    }
                });

                // audio buffer of size 128
                chunkData = ArrayChunk(audioFeatureValues, 128);

                // cal of number of blocks
                numBlocks = (audioFeatureValues.length - (blockLength - blockShift)) / blockShift;

                completeBuffer = new float[audioFeatureValues.length];

                // init of output1 and output2 regrading size of model output
                initOutput1();
                initOutput2();

                float[] part1 = new float[512];
                float[] temp = new float[512];


                for (int i = 0; i < numBlocks; i++) {

                    long preprocessStart = System.currentTimeMillis();

                    Log.d("data", "" + i);
                    long blockShiftStart = System.currentTimeMillis();

                    System.arraycopy(part1, 128, part1, 0, 384);
                    System.arraycopy(chunkData[i], 0, part1, 384, chunkData[i].length);
                    System.arraycopy(part1, 0, temp, 0, 512);

                    long blockShiftEnd = System.currentTimeMillis();
                    Log.d("time", "The process was running: Block shifting:- " + ((double) (blockShiftEnd - blockShiftStart) / 1000.0d) + "sec");
                    blockShiftTime.add((double) (blockShiftEnd - blockShiftStart) / 1000.0d);


                    long fftStart = System.currentTimeMillis();

                    // Forward Fourier Transform
                    float[] forwardFT = realForwardFT(temp);

                    long fftEnd = System.currentTimeMillis();
                    Log.d("time", "The process was running: FFT:- " + ((double) (fftEnd - fftStart) / 1000.0d) + "sec");
                    fft.add((double) (fftEnd - fftStart) / 1000.0d);

                    long absEstStart = System.currentTimeMillis();
                    //Calculate absolute
                    float[] absValues = getAbs(getPart("real", forwardFT), getPart("img", forwardFT));

                    // Calculate Phase Angle values for estimated complex values
                    float[] getPhaseValues = getPhaseAngle(getPart("real", forwardFT), getPart("img", forwardFT));

                    long absEstEnd = System.currentTimeMillis();
                    Log.d("time", "The process was running: ABS and PhaseAngle:- " + ((double) (absEstEnd - absEstStart) / 1000.0d) + "sec");
                    absEStTime.add((double) (absEstEnd - absEstStart) / 1000.0d);

                    // ArrayChunk will return 1d array to 2d and inBuffer is the input to model 1
                    inBuffer = ArrayChunk(absValues, 257);

                    long preProcessEnd = System.currentTimeMillis();
                    Log.d("time", "The process was running: preProcess 1 - " + ((double) (preProcessEnd - preprocessStart) / 1000.0d) + "sec");
                    pre1.add(((double) (preProcessEnd - preprocessStart) / 1000.0d));

                    final long model1start = System.currentTimeMillis();

                    // model1 process
                    initTflite1(TFLITE_FILE_1);


                    if (i == 0) {
                        feedTFLite1(inputShapeA(inBuffer), inputShapeB(null));
                    } else {
                        feedTFLite1(inputShapeA(inBuffer), inputShapeB("hashMapOutputB"));
                    }

                    final long model1end = System.currentTimeMillis();
                    Log.d("time", "The process was running: Model 1 - " + ((double) (model1end - model1start) / 1000.0d) + "sec");
                    model1time.add((double) (model1end - model1start) / 1000.0d);

                    final long preprocess2Start = System.currentTimeMillis();

                    //estimate values in 1d array
                    float[] estimatedComplexValues = estimatedComplex(absValues, outputOfModel1, getPhaseValues);

                    // Inverse Fourier Transform
                    float[] inverseFFT = realInverseFT(estimatedComplexValues);

                    //convert 1d array to 2d array of output of inverse fft
                    float[][] array2d = ArrayChunk(inverseFFT, 512);

                    // convert the 2d array to 3d array
                    float[][][] array3d = inputShapeC(array2d);


                    final long preprocess2End = System.currentTimeMillis();
                    Log.d("time", "The process was running: Preprocess 2 - " + ((double) (preprocess2End - preprocess2Start) / 1000.0d) + "sec");
                    pre2.add((double) (preprocess2End - preprocess2Start) / 1000.0d);

                    final long model2start = System.currentTimeMillis();

                    // model2 process
                    initTflite2(TFLITE_FILE_2);
                    if (i == 0) {
                        feedTFLite2(array3d, inputShapeD(null), i);
                    } else {
                        feedTFLite2(array3d, inputShapeD("hashMapOutputD"), i);
                    }

                    final long model2end = System.currentTimeMillis();
                    Log.d("time", "The process was running: Model 2- " + ((double) (model2end - model2start) / 1000.0d) + "sec");
                    model2time.add((double) (model2end - model2start) / 1000.0d);
                }

                total = 0.0;
                avg = 0.0;
                calculateTime(pre1, "pre1");
                total = 0.0;
                avg = 0.0;
                calculateTime(model1time, "model1time");
                total = 0.0;
                avg = 0.0;
                calculateTime(pre2, "pre2");
                total = 0.0;
                avg = 0.0;
                calculateTime(model2time, "model2time");
                total = 0.0;
                avg = 0.0;
                calculateTime(audioRead, "audioRead");
                total = 0.0;
                avg = 0.0;
                calculateTime(blockShiftTime, "blockShiftTime");
                total = 0.0;
                avg = 0.0;
                calculateTime(fft, "fft");
                total = 0.0;
                avg = 0.0;
                calculateTime(absEStTime, "absEStTime");
                //total = 0.0;
                //avg = 0.0;


                final long audioProcess = System.currentTimeMillis();

                // convert 1d float array to Byte array
                writeFloatToByte(completeBuffer);

                final long audioProcessEnd = System.currentTimeMillis();
                Log.d("total", "The process was running: Clean Audio File creation- " + ((double) (audioProcessEnd - audioProcess) / 1000.0d) + "sec");

                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        resultTextview.append("The process was running: Clean Audio File creation- " + ((double) (audioProcessEnd - audioProcess) / 1000.0d) + "sec\n");
                    }
                });

                //Log.d("dd", "success" + outputBuffer);
            } catch (Exception e) {
                Log.e(TAG + " Exception", e.getMessage());
            }

            final long totalend = System.currentTimeMillis();
            Log.d("time", "The program was running: Total :" + ((double) (totalend - totalstart) / 1000.0d) + "sec \n ***************************************************");

            stringBuilder.append("The program was running: Total :" + ((double) (totalend - totalstart) / 1000.0d) + "sec \n***********************************************\n");



            return null;
        }

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            progressBar.setVisibility(View.VISIBLE);

        }

        @Override
        protected void onPostExecute(Void unused) {
            super.onPostExecute(unused);
            progressBar.setVisibility(View.GONE);
            resultTextview.setText(stringBuilder.toString());
            Toast.makeText(getApplicationContext(), "Completed", Toast.LENGTH_SHORT).show();
        }
    }

    private void calculateTime(ArrayList<Double> arrayList, String tag) {

            for (int i = 0; i < arrayList.size(); i++) {
                total += arrayList.get(i);
            }
            avg = total / arrayList.size();
            Log.d("total", "The total is of " + tag + ":: " + String.format("%.3f", total) + " sec");
            Log.d("time","The Average IS of "+ tag +" "+String.format("%.6f", avg)+" sec");

            stringBuilder.append("The total is of " + tag + ":: " + String.format("%.3f", total) + " sec \n");


    }
}



