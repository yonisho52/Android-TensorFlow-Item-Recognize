package com.example.tensorflow_item_recognize;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.tensorflow_item_recognize.ml.MobilenetV110224Quant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {


    private static final int RESULT_LOAD_IMG = 0;
    private ImageView image_view;
    private ArrayList<String> arr;
    private TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        arr = new ArrayList<String>();

        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(getAssets().open("label.txt") , "UTF-8"));
            String mLine=null;
            while( (mLine = reader.readLine()) != null){
                arr.add(mLine);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void funcSelect(View view) {

        Intent photoPicker = new Intent(Intent.ACTION_PICK);
        photoPicker.setType("image/*");
        startActivityForResult(photoPicker , RESULT_LOAD_IMG);

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        image_view = findViewById(R.id.imageView);

        if ( resultCode == RESULT_OK)
        {
            final Uri imageUri = data.getData();
            final InputStream imageStream;
            try {
                imageStream = getContentResolver().openInputStream(imageUri);
                final Bitmap selectImageBitmap = BitmapFactory.decodeStream(imageStream);
                image_view.setImageBitmap(selectImageBitmap);

            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }


        }
        else
        {
            Toast.makeText(this , "you havent picked image", Toast.LENGTH_LONG).show();
        }
    }

    public void func_detect(View view) {

        textView = findViewById(R.id.textViewResult);
        Bitmap bm = ((BitmapDrawable)image_view.getDrawable()).getBitmap();
        Bitmap resize = Bitmap.createScaledBitmap(bm , 224 , 224 , true);

        try {
            TensorImage selectImage = TensorImage.fromBitmap(resize);

            MobilenetV110224Quant model = MobilenetV110224Quant.newInstance(this);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);

            ByteBuffer byteBuffer = selectImage.getBuffer();

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            MobilenetV110224Quant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

//            int maxIndex = getMax( outputFeature0.getFloatArray());
//
//            textView.setText(arr.get(maxIndex));

            getMaxMap(outputFeature0.getFloatArray());


            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

    }

    public int getMax (float[] fa )
    {
        int index = 0;
        float max =0;

        for(int i=0 ; i< fa.length ; i++)
        {
            if(fa[i] > max )
            {
                index = i;
                max = fa[i];

            }
        }
        Log.d("MAXIMUM" , String.valueOf(fa[index]));
        return index;
    }

    public void getMaxMap (float[] fa)
    {

        for(int j=0;j<3;j++)
        {
            int index = 0;
            float max =0;

            for(int i=0 ; i< fa.length ; i++)
            {
                if(fa[i] > max )
                {
                    index = i;
                    max = fa[i];
                }
            }
            float num = fa[index]/255*100;
            textView.append(arr.get(index) + " : " + new DecimalFormat("##.##").format(num) +"\n");
            fa[index] = 0;
        }


    }

}