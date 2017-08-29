package com.ppltech.predictor.model;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;


/**
 * Class to load the raw data(CSV file) and converts into list of
 * model(FacebookStats) suitable for training and validation
 * 
 * @author arun
 *
 */
public class FileLoader {

	public List<FacebookStats> loadData(String fileName) {
		List<FacebookStats> list = new ArrayList<FacebookStats>();
		try {
			
			BufferedReader br = new BufferedReader(new FileReader(fileName));

			String[] fieldNames = br.readLine().split(",");
			String sCurrentLine = null;
			while ((sCurrentLine = br.readLine()) != null) {
				list.add(new FacebookStats(fieldNames, sCurrentLine.split(",")));
			}
			br.close();
			
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
		return list;

	}

}
