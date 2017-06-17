package com.example.yuya.myapplication;

import android.app.ProgressDialog;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.ListAdapter;
import android.widget.ListView;
import android.widget.SimpleAdapter;
import android.widget.Toast;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.test2;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private String TAG = MainActivity.class.getSimpleName();

    private ProgressDialog pDialog;
    private ListView lv;

    // URL to get contacts JSON
    private static String url = "http://newsapi.org/v1/articles?source=bbc-news&sortBy=top&apiKey=144dd64c47c541738c17bec4a2656295";
    //private static String url = "http://api.androidhive.info/contacts/";

    ArrayList<HashMap<String, String>> contactList;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        contactList = new ArrayList<>();

        lv = (ListView) findViewById(R.id.listView);

        new GetContacts().execute();
    }

    /**
     * Async task class to get json by making HTTP call
     */
    private class GetContacts extends AsyncTask<Void, Void, Void> {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            // Showing progress dialog
            pDialog = new ProgressDialog(MainActivity.this);
            pDialog.setMessage("Please wait...");
            pDialog.setCancelable(false);
            pDialog.show();

        }

        @Override
        protected Void doInBackground(Void... arg0) {
            HttpHandler sh = new HttpHandler();

            // Making a request to url and getting response
            String jsonStr = sh.makeServiceCall(url);

            Log.e(TAG, "Response from url: " + jsonStr);

            if (jsonStr != null) {
                try {
                    JSONObject jsonObj = new JSONObject(jsonStr);

                    // Getting JSON Array node
                    JSONArray contacts = jsonObj.getJSONArray("articles");

                    // looping through All Contacts
                    for (int i = 0; i < contacts.length(); i++) {
                        JSONObject c = contacts.getJSONObject(i);

                        String id = c.getString("author");
                        String name = c.getString("title");
                        String email = c.getString("description");
                        String address = c.getString("url");
                        String gender = c.getString("urlToImage");
                        String mobile = c.getString("publishedAt");

                        // Phone node is JSON Object
                        //JSONObject phone = c.getJSONObject("phone");
                        //String mobile = phone.getString("mobile");
                        //String home = phone.getString("home");
                        //String office = phone.getString("office");

                        // tmp hash map for single contact
                        HashMap<String, String> contact = new HashMap<>();

                        // adding each child node to HashMap key => value
                        contact.put("id", id);
                        contact.put("name", name);
                        contact.put("email", email);
                        contact.put("mobile", mobile);

                        // adding contact to contact list
                        contactList.add(contact);
                    }
                } catch (final JSONException e) {
                    Log.e(TAG, "Json parsing error: " + e.getMessage());
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(getApplicationContext(),
                                    "Json parsing error: " + e.getMessage(),
                                    Toast.LENGTH_LONG)
                                    .show();
                        }
                    });

                }
            } else {
                Log.e(TAG, "Couldn't get json from server.");
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(getApplicationContext(),
                                "Couldn't get json from server. Check LogCat for possible errors!",
                                Toast.LENGTH_LONG)
                                .show();
                    }
                });

            }

            return null;
        }

        @Override
        protected void onPostExecute(Void result) {
            super.onPostExecute(result);
            // Dismiss the progress dialog
            if (pDialog.isShowing())
                pDialog.dismiss();
            /**
             * Updating parsed JSON data into ListView
             * */
            ListAdapter adapter = new SimpleAdapter(
                    MainActivity.this, contactList,
                    R.layout.list_item, new String[]{"name", "email",
                    "mobile"}, new int[]{R.id.name,
                    R.id.email, R.id.mobile});

            lv.setAdapter(adapter);
        }

    }

    private class test{
        private final Logger log = LoggerFactory.getLogger(test2.class);

        public void main(String[] args) throws Exception {

            ParagraphVectors paragraphVectors;
            LabelAwareIterator iterator;
            TokenizerFactory tokenizerFactory;
            paragraphVectors= readParagraphVectors("pathToSaveModel3");

            ClassPathResource resource = new ClassPathResource("paravec/labeled");
            // build a iterator for our dataset
            iterator = new FileLabelAwareIterator.Builder()
                    .addSourceFolder(resource.getFile())
                    .build();

            tokenizerFactory = new DefaultTokenizerFactory();
            tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

            ClassPathResource unClassifiedResource = new ClassPathResource("paravec/unlabeled");
            FileLabelAwareIterator unClassifiedIterator = new FileLabelAwareIterator.Builder()
                    .addSourceFolder(unClassifiedResource.getFile())
                    .build();

            MeansBuilder meansBuilder = new MeansBuilder(
                    (InMemoryLookupTable<VocabWord>)paragraphVectors.getLookupTable(),
                    tokenizerFactory);
            LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
                    (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

            while (unClassifiedIterator.hasNextDocument()) {
                LabelledDocument document = unClassifiedIterator.nextDocument();
                INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
                List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

         /*
          please note, document.getLabel() is used just to show which document we're looking at now,
          as a substitute for printing out the whole document name.
          So, labels on these two documents are used like titles,
          just to visualize our classification done properly
         */
                log.info("Document '" + document.getLabel() + "' falls into the following categories: ");
                for (Pair<String, Double> score: scores) {
                    log.info("        " + score.getFirst() + ": " + score.getSecond());
                }
            }
        }
    }
}
