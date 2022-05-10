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

    private MappedByteBuffer tfLiteModel;
    private Interpreter tfLite;
    private Spinner audioClipSpinner;
    private Button transcribeButton;
    private Button playAudioButton;
    private TextView resultTextview;
    private String wavFilename;
    private MediaPlayer mediaPlayer = new MediaPlayer();
    private final static String TAG = "MainActivity";
    private final static int SAMPLE_RATE = 16000;
    private final static int DEFAULT_AUDIO_DURATION = -1;
    private final static String[] WAV_FILENAMES = {"audio_clip_4.wav"};
    private final static String TFLITE_FILE = "model_1.tflite";

    float[][] chunkData;
    float[][][] inputShape1;
    float[][][][] inputShape2;
    Map<Integer, Object> outputMap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        JLibrosa jLibrosa = new JLibrosa();
        audioClipSpinner = findViewById(R.id.audio_clip_spinner);
        ArrayAdapter<String> adapter = new ArrayAdapter<String>(MainActivity.this,
                android.R.layout.simple_spinner_item, WAV_FILENAMES);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        audioClipSpinner.setAdapter(adapter);
        audioClipSpinner.setOnItemSelectedListener(this);
        playAudioButton = findViewById(R.id.play);
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

        transcribeButton = findViewById(R.id.recognize);
        resultTextview = findViewById(R.id.result);
        transcribeButton.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                try {
                        initOutput();
                        float a = 1.0f;
                        float[] actualBuffer = new float[512];
                        //Arrays.fill(actualBuffer, a+1f);
                        for (int i = 0; i < actualBuffer.length; i++) {
                            actualBuffer[i] = a++;
                        }
                        float[] part1 = Arrays.copyOfRange(actualBuffer, 127, 511);
                        float[] part2 = Arrays.copyOfRange(actualBuffer, 383, 511);
                        float[] newBuffer = new float[part1.length + part2.length];
                        System.arraycopy(part1, 0, newBuffer, 0, part1.length);
                        System.arraycopy(part2, 0, newBuffer, part1.length, part2.length);
                        Log.d(TAG, Arrays.toString(newBuffer));
                        float[] audioFeatureValues = jLibrosa.loadAndRead(copyWavFileToCache(wavFilename), SAMPLE_RATE, DEFAULT_AUDIO_DURATION);
                        chunkData = ArrayChunk(audioFeatureValues, 512);
                        // Forward Fourier Transform
                        realForwardFT();
                        // Inverse Forier Transform
                        realInverseFT();
                        //Calculate absolute
                        float[] absValues = getAbs(getPart("real"), getPart("img"));
                        float[] getPhaseValues = getPhaseAngle(getPart("real"), getPart("img"));
                        chunkData = ArrayChunk(absValues, 256);
                        initTflite();
                        feedTFLite(inputShapeA(chunkData), inputShapeB(null));
                } catch (Exception e) {
                    Log.e(TAG + " Exception", e.getMessage());
                }
            }
        });
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

    private float[][][] inputShapeA(float[][] input){
        inputShape1 = new float[1][1][257];
        for (int i = 0; i < input[0].length; i++) {
            inputShape1[0][0][i] = (input[0][i]);
        }
        if(input[0].length == 256){
            inputShape1[0][0][256] = input[0][255];
        }
        return inputShape1;
    }

    private float[][][][] inputShapeB(float[][] input){
        if(input == null){
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

    public void initOutput(){
        IntBuffer outputBuffer = IntBuffer.allocate(2000);
        outputMap = new HashMap<>();
        float[][][] out1 = new float[1][1][257];
        float[][][][] out2 = new float[1][2][128][2];
        outputMap.put(0, out1);
        outputMap.put(1, out2);
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

    private void initTflite() throws IOException {
        tfLiteModel = loadModelFile(getAssets(), TFLITE_FILE);
        Interpreter.Options tfLiteOptions = new Interpreter.Options();
        tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
    }

    private void feedTFLite(float[][][] f1, float[][][][] f2){
        Object[] inputArray = {f1, f2};
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        processOutput(outputMap);
        Log.d("XXX", "Success");
    }

    private ArrayList<Float> getPart(String part){
        ArrayList<Float> real = new ArrayList<>();
        ArrayList<Float> img = new ArrayList<>();
        for (int i = 0; i < chunkData[0].length; i++) {
            if (i % 2 == 0) {
                //Even
                real.add(chunkData[0][i]);
            } else {
                //Odd
                img.add(chunkData[0][i]);
            }
        }
        if(part.equals("real")){
            return real;
        }else{
            return img;
        }
    }

    private float[] realForwardFT(){
        FloatFFT_1D floatFFT_1D = new FloatFFT_1D(chunkData[0].length);
        floatFFT_1D.realForward(chunkData[0]);
        return chunkData[0];
    }

    private float[] realInverseFT(){
        FloatFFT_1D floatFFT_1D = new FloatFFT_1D(chunkData[0].length);
        floatFFT_1D.realInverse(chunkData[0], true);
        return chunkData[0];
    }

//    private void processOutput(Map<Integer, Object> outputMap){
//        f1 = outputMap.get(0);
//    }

}