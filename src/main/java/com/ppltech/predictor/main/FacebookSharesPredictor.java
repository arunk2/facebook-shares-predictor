package com.ppltech.predictor.main;

import java.util.Collections;
import java.util.List;

import org.apache.mahout.classifier.evaluation.Auc;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.Vector;

import com.ppltech.predictor.model.FacebookStats;
import com.ppltech.predictor.model.FileLoader;

/**
 * Main class for Facebook share buckets prediction.
 * We use Logistic regression (multi-class) algorithm for
 * training and prediction.
 * 
 * We had 4 categories of the content influencing the 
 * popularity and hence shares.
 * 
 * @author arun
 *
 */
public class FacebookSharesPredictor {

	public static final int NUM_CATEGORIES = 4;

	public static void main(String[] args) throws Exception {
		FileLoader loader = new FileLoader();
		List<FacebookStats> calls = loader.loadData("src/main/resources/facebook_stats.csv");

		double heldOutPercentage = 0.10;

		Collections.shuffle(calls);
		int cutoff = (int) (heldOutPercentage * calls.size());
		List<FacebookStats> test = calls.subList(0, cutoff);
		List<FacebookStats> train = calls.subList(cutoff, calls.size());

		OnlineLogisticRegression lr = new OnlineLogisticRegression(
				NUM_CATEGORIES, FacebookStats.FEATURES, new L1())
				.learningRate(1).alpha(1).lambda(0.000001).stepOffset(10000)
				.decayExponent(0.2);

		for (int pass = 0; pass < 500; pass++) {
			for (FacebookStats observation : train) {
				lr.train(observation.getTarget(), observation.asVector());
			}
			if (pass % 5 == 0) {
				Auc eval = new Auc(.25);
				for (FacebookStats testCall : test) {
					Vector classify = lr.classifyFull(testCall.asVector());

					if (testCall.getTarget() == classify.maxValueIndex())
						eval.add(1, classify.maxValue());
					else
						eval.add(0, classify.maxValue());
				}

				System.out.printf("%d, %.4f, %.4f\n", pass, lr.currentLearningRate(), eval.auc());
			}
		}

		// Classify the unknown
		String fields = "DateTime,ReachCategory,Cat1,Cat2,Cat3,Cat4,Cat5";
		String values = "02/06/16 18:14,400-600K,Veg,Starter,Side dish,,";

		FacebookStats tc = new FacebookStats(fields.split(","),
				values.split(","));
		Vector classify = lr.classifyFull(tc.asVector());
		System.out.println(classify);

	}
	
}
