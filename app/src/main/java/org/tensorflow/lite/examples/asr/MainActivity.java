package org.tensorflow.lite.examples.asr;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.MediaPlayer;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import com.jlibrosa.audio.JLibrosa;

import org.jtransforms.fft.FloatFFT_1D;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    private final static String TAG = "MainActivity";

    private Spinner audioClipSpinner;
    private Button transcribeButton;
    private Button playAudioButton;
    private TextView resultTextview;

    private MediaPlayer mediaPlayer = new MediaPlayer();

    private final static int SAMPLE_RATE = 16000;
    private final static int DEFAULT_AUDIO_DURATION = -1;

    private final static String[] WAV_FILENAMES = {"audio_clip_4.wav"};
    private final static String TFLITE_FILE_1 = "model_1.tflite";
    private final static String TFLITE_FILE_2 = "model_2.tflite";
    private String wavFilename;

    float[][] chunkData;
    float[][] inBuffer;
    float[][][] inputShape1;
    float[][][][] inputShape2;

    // final outputs from models
    float[] outputOfModel1, outputOfModel2;

    Map<Integer, Object> outputMap1, outputMap2;

    int numBlocks;
    int blockShift = 128;
    int blockLength = 512;

    private MappedByteBuffer tfLiteModel1, tfLiteModel2;
    private Interpreter tfLite1, tfLite2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        JLibrosa jLibrosa = new JLibrosa();

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
                try {
                    // full audio buffer
                    float[] audioFeatureValues = jLibrosa.loadAndRead(copyWavFileToCache(wavFilename), SAMPLE_RATE, DEFAULT_AUDIO_DURATION);

                    // audio buffer of size 128
                    chunkData = ArrayChunk(audioFeatureValues, 128);

                    // cal of number of blocks
                    numBlocks = (audioFeatureValues.length - (blockLength - blockShift)) / blockShift;

                    // init of output1 and output2 regrading size of model output
                    initOutput1();
                    initOutput2();

                    //float a = 1.0f;
                    //float[] actualBuffer = new float[512];
                    //Arrays.fill(actualBuffer, 0.0f);

                    float part1[] = new float[512];

                    for (int i = 0; i < numBlocks; i++) {

                        System.arraycopy(part1, 128, part1, 0, 384);
                        System.arraycopy(chunkData[i], 0, part1, 384, chunkData[i].length);

                        // Forward Fourier Transform
                        float[] forwardFT = realForwardFT(part1);


                        //Calculate absolute
                        float[] absValues = getAbs(getPart("real", forwardFT), getPart("img", forwardFT));
                        float[] getPhaseValues = getPhaseAngle(getPart("real", forwardFT), getPart("img", forwardFT));

                        //chunkData = ArrayChunk(absValues, 256); // TODO: Discuss with gaurav


                        // model process
                        initTflite1(TFLITE_FILE_1);
                        feedTFLite1(inputShapeA(chunkData), inputShapeB(null));

                        //estimate values in 1d array
                        float[] forInverseFFT = estimatedComplex(absValues, outputOfModel1, getPhaseValues);

                        // Inverse Fourier Transform
                        float[] inverseFFT = realInverseFT(forInverseFFT);

                        //convert 1d array to 2d array of output of inverse fft
                        float[][] array2d = ArrayChunk(inverseFFT, 512);

                        // convert the 2d array to 3d array
                        float[][][] array3d = inputShapeC(array2d);

                        initTflite2(TFLITE_FILE_2);
                        feedTFLite2(array3d, inputShapeB(null));

                    }

                } catch (Exception e) {
                    Log.e(TAG + " Exception", e.getMessage());
                }
            }
        });
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
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View v, int position, long id) {
        wavFilename = WAV_FILENAMES[position];
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {
    }

    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
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

    private float[][][][] inputShapeB(float[][] input) {
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
            return inputShape2;
        }
        return null;
    }

    public void initOutput1() {
        IntBuffer outputBuffer = IntBuffer.allocate(2000);
        outputMap1 = new HashMap<>();
        float[][][] out1 = new float[1][1][257];
        float[][][][] out2 = new float[1][2][128][2];
        outputMap1.put(0, out1);
        outputMap1.put(1, out2);
    }

    public void initOutput2() {
        IntBuffer outputBuffer = IntBuffer.allocate(2000);
        outputMap2 = new HashMap<>();
        float[][][] out1 = new float[1][1][512];
        float[][][][] out2 = new float[1][2][128][2];
        outputMap2.put(0, out1);
        outputMap2.put(1, out2);
    }

    public float[] getPhaseAngle(ArrayList<Float> real, ArrayList<Float> img) {
        float[] phaseAngle = new float[real.size()];
        for (int i = 0; i < real.size(); i++) {
            phaseAngle[i] = (float) Math.atan(real.get(i) / img.get(i));
        }
        return phaseAngle;
    }

    private float[] getAbs(ArrayList<Float> real, ArrayList<Float> img) {
        float[] abs = new float[real.size()];
        for (int i = 0; i < real.size(); i++) {
            abs[i] = (float) Math.sqrt((real.get(i) * real.get(i)) + (img.get(i) * img.get(i)));
        }
        return abs;
    }

    private void initTflite1(String model) throws IOException {
        tfLiteModel1 = loadModelFile(getAssets(), model);
        Interpreter.Options tfLiteOptions = new Interpreter.Options();
        tfLite1 = new Interpreter(tfLiteModel1, tfLiteOptions);
    }

    private void feedTFLite1(float[][][] f1, float[][][][] f2) {
        Object[] inputArray = {f1, f2};
        tfLite1.runForMultipleInputsOutputs(inputArray, outputMap1);
        processOutput1(outputMap1);
        Log.d("XXX", "Success");
    }

    private void initTflite2(String model) throws IOException {
        tfLiteModel2 = loadModelFile(getAssets(), model);
        Interpreter.Options tfLiteOptions = new Interpreter.Options();
        tfLite2 = new Interpreter(tfLiteModel2, tfLiteOptions);
    }

    private void feedTFLite2(float[][][] f1, float[][][][] f2) {
        Object[] inputArray = {f1, f2};
        tfLite2.runForMultipleInputsOutputs(inputArray, outputMap2);
        processOutput2(outputMap2);
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

    private void processOutput1(Map<Integer, Object> outputMap) {
        float[][][] hashMapOutput = (float[][][]) outputMap.get(0);
        outputOfModel1 = hashMapOutput[0][0];
        Log.d("cc", "" + outputOfModel1);
    }

    private void processOutput2(Map<Integer, Object> outputMap) {
        float[][][] hashMapOutput = (float[][][]) outputMap.get(0);
        outputOfModel2 = hashMapOutput[0][0];
        Log.d("cc", "" + outputOfModel2);
    }

}