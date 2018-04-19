package com.example.ram.loginform1;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    private EditText username;
    private EditText password;
    private TextView attempts;
    private Button login;
    private int counter=5;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        username= (EditText)findViewById(R.id.usr);
        password= (EditText)findViewById(R.id.psw);
        attempts=(TextView)findViewById(R.id.atmpt);
        login=(Button)findViewById(R.id.logn);

        //attempts.setText("No of Attempts Remain : 5");

        login.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                validate(username.getText().toString(),password.getText().toString());
            }
        });


    }
    private void validate(String usr,String psw){

        if( ( usr.equals( "admin") || usr.equals( "ADMIN") ) &&  psw.equals("123456") )
        {
            Intent i=new Intent(MainActivity.this,SecondActivity.class);
            startActivity(i);
        }
        else{
            attempts.setText("Incorrect Username or Password");
            counter--;
           /* attempts.setText("No of Attempts Remain : " + String.valueOf(counter));
            if(counter==0)
               login.setEnabled(false);
               */

        }
    }
}
