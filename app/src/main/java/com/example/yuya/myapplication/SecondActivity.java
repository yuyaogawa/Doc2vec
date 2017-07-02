package com.example.yuya.myapplication;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.webkit.WebView;

/**
 * Created by yuya on 6/27/2017.
 */
public class SecondActivity extends Activity {

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_second);

        Intent data = getIntent();
        Bundle extras = data.getExtras();
        String disp_pict = extras != null ? extras.getString("SELECTED_PICT") : "";

        WebView webview = (WebView) findViewById(R.id.webView1);

        // 表示するWEBサイトの設定
        if (disp_pict.equals("aaa")) {
            webview.loadUrl("http://www.bbc.com/news/world-us-canada-40413563");
        } else if (disp_pict.equals("Banana")) {
            webview.loadUrl("http://aaa.aaa.aa");
        } else {
            webview.loadUrl(disp_pict);
        }
    }


    class ButtonClickListener implements View.OnClickListener {
        public void onClick(View v) {
            finish();
        }
    }
}
