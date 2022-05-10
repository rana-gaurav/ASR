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

    private int num_blocks;
    private int block_shift = 128;
    private int block_length = 512;

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
                    // filling empty buffer of 512 with increment data like 0 to 511
                    float a = 1.0f;
                    float[] actualBuffer = new float[512];
                    //Arrays.fill(actualBuffer, a+1f);
                    for (int i = 0; i < actualBuffer.length; i++) {
                        actualBuffer[i] = a++;
                    }

                    // split into two parts of actually buffer
                    float part1[] = Arrays.copyOfRange(actualBuffer, 127, 511);
                    float part2[] = Arrays.copyOfRange(actualBuffer, 383, 511);

                    // create a new buffer and block shifting with length of 128 to 512 to starting of array and at last 128 add the audio data
                    float[] newBuffer = new float[part1.length + part2.length];
                    System.arraycopy(part1, 0, newBuffer, 0, part1.length);
                    System.arraycopy(part2, 0, newBuffer, part1.length, part2.length);
                    Log.d(TAG, Arrays.toString(newBuffer));

                    // reading the wave file using Jlibrosa in float array
                    float audioFeatureValues[] = jLibrosa.loadAndRead(copyWavFileToCache(wavFilename), SAMPLE_RATE, DEFAULT_AUDIO_DURATION);

                    num_blocks=(audioFeatureValues.length-(block_length-block_shift))/block_shift;

                    // splitting the audio data into 512 size of each
                    float chunkData[][] = ArrayChunk(audioFeatureValues, 512);

                    // Fourier Transform
                    FloatFFT_1D floatFFT_1D = new FloatFFT_1D(chunkData[0].length);
                    floatFFT_1D.realForward(chunkData[0]);
                    //floatFFT_1D.realInverse(chunkData[0], true);

                    //separation of real and imaginary values of FFT complex output
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

                    // Abs Values
                    float[] absValues = getAbs(real, img);
                    Log.d("abs", Arrays.toString(absValues));

                    // Phase values
                    float[] phaseValues = getPhaseAngle(real, img);
                    Log.d("phaseValues", Arrays.toString(phaseValues));


                    IntBuffer outputBuffer = IntBuffer.allocate(2000);
                    Map<Integer, Object> outputMap = new HashMap<>();
                    float[][][] out1 = new float[1][1][257];
                    float[][][][] out2 = new float[1][2][128][2];
                    outputMap.put(0, out1);
                    outputMap.put(1, out2);

                    tfLiteModel = loadModelFile(getAssets(), TFLITE_FILE);
                    Interpreter.Options tfLiteOptions = new Interpreter.Options();
                    tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
                    // Input 1
                    float[][][] f1 = new float[1][1][257];
                    for (int i = 0; i < chunkData[0].length; i++) {
                        f1[0][0][i] = (chunkData[0][i]);
                    }

                    // Input 2 initial array
                    float[][][][] f2 = {{{{0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
                            {0.0f, 0.0f},
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
                    Object[] inputArray = {f1, f2};
//                        tfLite.resizeInput(0, new int[] {1,1,257});
//                        tfLite.resizeInput(1, new int[] {1,2,128,2});
                    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
                    Log.d(TAG, "Output Success");
                    int outputSize = tfLite.getOutputTensor(0).shape()[0];
                    int[] outputArray = new int[outputSize];
                    outputBuffer.rewind();
                    outputBuffer.get(outputArray);

                    estimatedComplex(img,outputArray,phaseValues);

                    //tfLite.run(byteBuffer, outputBuffer);

                    //                    StringBuilder finalResult = new StringBuilder();
                    //                    for (int i=0; i < outputSize; i++) {
                    //                        char c = (char) outputArray[i];
                    //                        if (outputArray[i] != 0) {
                    //                            finalResult.append((char) outputArray[i]);
                    //                        }
                    //                    }
                    //                    resultTextview.setText(finalResult.toString());
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

    byte[] toPrimitives(Byte[] oBytes) {
        byte[] bytes = new byte[oBytes.length];

        for (int i = 0; i < oBytes.length; i++) {
            bytes[i] = oBytes[i];
        }

        return bytes;
    }

    public static double[] convertFloatsToDoubles(float[] input) {
        if (input == null) {
            return null; // Or throw an exception - your choice
        }
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    public float[] getAbs(float[] re, float[] img) {

        float[] abs = new float[re.length];
        for (int i = 0; i < re.length; i++) {
            abs[i] = (float) Math.sqrt((re[i] * re[i]) + (img[i] + img[i]));
        }
        return abs;
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
        for (int i = 0; i <= real.size(); i++) {
            abs[i] = (float) Math.sqrt((real.get(i) * real.get(i)) + (img.get(i) * img.get(i)));
        }
        return abs;
    }

    private void estimatedComplex(ArrayList<Float> img, int[] outputOfModel, float[] phaseAngle){
        float[][][] estimatedValues=new float[1][1][img.size()];
        for (int i=0;i<img.size();i++) {
            estimatedValues[0][0][i] = (float) (img.get(i) *outputOfModel[i] * Math.exp(1*phaseAngle[i]));
        }
        Log.d("estimated complex",Arrays.toString(estimatedValues));



    }


}