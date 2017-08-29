package com.ppltech.predictor.model;

import java.util.HashMap;
import java.util.Map;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ConstantValueEncoder;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

/**
 * Create a model for training the logistic-regression algorithm.
 * Following line gives heading and the sample entry. 
 * All the 5 categories(cat1, cat2, cat3, cat4, cat5 - keywords) are considered as plain words
 * (from a set of limited words) and hence directly encoded as features
 * 
 * Example:
 * 
 * DateTime			ReachCategory	Cat1	Cat2	Cat3		Cat4			Cat5
 * 01/13/16 7:49pm	0-200K			Veg		Snacks	deepFried	Andhraspecial

 * @author arun
 *
 */

public class FacebookStats {
	public static final int FEATURES = 500;
	private static final ConstantValueEncoder interceptEncoder = new ConstantValueEncoder(
			"intercept");
	private static final FeatureVectorEncoder featureEncoder = new StaticWordValueEncoder(
			"feature");

	private RandomAccessSparseVector vector;

	private Map<String, String> fields = new HashMap<String, String>();

	public FacebookStats(String[] fieldNames, String[] values) {
		vector = new RandomAccessSparseVector(FEATURES);

		interceptEncoder.addToVector("1", vector);

		int i = 0;
		for (String fieldValue : values) {
			String name = fieldNames[i++];
			fields.put(name, fieldValue);

			if (fieldValue == null || fieldValue.equals("")) {
				continue;
			}
			switch (name) {

			case "Cat1":
			case "Cat2":
			case "Cat3":
			case "Cat4":
			case "Cat5":
				featureEncoder.addToVector(fieldValue, 1, vector);
				break;

			case "ReachCategory":
			case "DateTime":
				break; // ignore these fields them

			default:
				throw new IllegalArgumentException(String.format(
						"Bad field name: %s", name));
				
			}
		}
	}

	public Vector asVector() {
		return vector;
	}

	public int getTarget() {
		Integer reachBucket = 0;
		String reach = fields.get("ReachCategory");

		switch (reach) {
		case "0-200K":
			reachBucket = 0;
			break;
		case "200-400K":
			reachBucket = 0;
			break;
		case "400-600K":
			reachBucket = 1;
			break;
		case "600-800K":
			reachBucket = 1;
			break;
		case "800-1000K":
			reachBucket = 2;
			break;
		case "1000-1200K":
			reachBucket = 2;
			break;
		case "2000-2200K":
			reachBucket = 3;
			break;
		}
		return reachBucket;
	}

	public Map<String, String> getFields() {
		return fields;
	}
}
