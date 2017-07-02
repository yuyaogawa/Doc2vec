package org.deeplearning4j;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.tools.LabelSeeker;
import org.deeplearning4j.tools.MeansBuilder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.List;
import static org.deeplearning4j.models.embeddings.loader.WordVectorSerializer.readParagraphVectors;

/**
 * Created by yuya on 6/17/2017.
 */
public class test2 {

    private static final Logger log = LoggerFactory.getLogger(test2.class);

    public static void main(String[] args) throws Exception {

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
