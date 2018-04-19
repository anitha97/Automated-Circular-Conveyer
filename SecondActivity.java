package com.example.ram.loginform1;

import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;

import static android.R.id.message;

public class SecondActivity extends AppCompatActivity {



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_second);

    }

    public void start_send_data(View v) {


        BackgroundTask b1 = new BackgroundTask();
        b1.execute("1");
    }

   /* public void stop_send_data(View v){
        BackgroundTask b2=new BackgroundTask();
        b2.execute("0");
    }*/


    class BackgroundTask extends AsyncTask<String, Void, Void> {

        Socket s;
        PrintWriter pw;

        @Override
        protected Void doInBackground(String... voids) {
            String message=voids[0];
            try {

                s = new Socket("202.91.86.218", 80);
                pw = new PrintWriter(s.getOutputStream());
                pw.write(message);
                pw.flush();
                pw.close();
                s.close();

            } catch (IOException e) {
                e.printStackTrace();
            }

            return null;

        }
    }
}