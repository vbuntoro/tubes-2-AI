/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubesai.weka;

import com.mysql.jdbc.Driver;
import java.io.File;
import java.sql.DriverManager;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javax.faces.bean.ManagedBean;
import javax.faces.bean.ApplicationScoped;
import javax.servlet.http.Part;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.WordTokenizer;
import weka.experiment.InstanceQuery;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 *
 * @author Asus
 */
@ManagedBean(name="Driver",eager=true)
@ApplicationScoped
public class DriverWEKA {

    /**
     * Creates a new instance of Driver
     * 
     * @throws java.lang.ClassNotFoundException
     * @throws java.lang.Exception
     */
    
    private final InstanceQuery loader;
    private final Instances data;
    private Instances datafilter;
    private final NominalToString nos;
    private StringToWordVector swv;
    private final WordTokenizer wt;
    private MultiFilter mfilter;
    private Part part1;
    private Part part2;
    private final List<String> algoList;
    private Classifier cls;
    private String algoSelection;
    private String outputText;
    private final static String[] saveclf = {"bayes","smo"};
    private boolean is_eval;
    private final String homepath;

    public boolean isIs_eval() {
        return is_eval;
    }

    public void setIs_eval(boolean is_eval) {
        this.is_eval = is_eval;
    }

    public Part getPart1() {
        return part1;
    }

    public void setPart1(Part part1) {
        this.part1 = part1;
    }

    public Part getPart2() {
        return part2;
    }

    public void setPart2(Part part2) {
        this.part2 = part2;
    }

    public List<String> getAlgoList() {
        return algoList;
    }

    public String getAlgoSelection() {
        return algoSelection;
    }

    public void setAlgoSelection(String algoSelection) {
        this.algoSelection = algoSelection;
    }

    public String getOutputText() {
        return outputText;
    }
    
    private String getFileName(Part part) {
		final String partHeader = part.getHeader("content-disposition");
		System.out.println("***** partHeader: " + partHeader);
		for (String content : part.getHeader("content-disposition").split(";")) {
			if (content.trim().startsWith("filename")) {
				return content.substring(content.indexOf('=') + 1).trim()
						.replace("\"", "");
			}
		}
		return null;
    }
    
    public DriverWEKA() throws Exception
    {
        DriverManager.registerDriver((Driver)Class.forName("com.mysql.jdbc.Driver").newInstance());
        loader = new InstanceQuery();
        loader.setUsername("root");
        loader.setPassword("");
        loader.setQuery("SELECT `artikel`.JUDUL,`artikel`.FULL_TEXT,`kategori`.LABEL\n" +
                        "FROM `artikel`\n" +
                        "NATURAL JOIN `artikel_kategori_verified`\n" +
                        "NATURAL JOIN `kategori`");
        data = loader.retrieveInstances();
        data.setClassIndex(data.numAttributes() - 1);
        System.out.println("Loaded database");
        System.out.println(data.toSummaryString());
        nos = new NominalToString();
        nos.setAttributeIndexes("1-2");
        swv = new StringToWordVector();
        swv.setAttributeIndices("1-2");
        wt = new WordTokenizer();
        wt.setDelimiters(" \r\t\n.,;:\'\"()?!-><#$\\%&*+/@^_=[]{}|`~0123456789");
        swv.setTokenizer(wt);
        swv.setLowerCaseTokens(true);
        swv.setOutputWordCounts(true);
        algoList = new ArrayList<>();
        algoList.add("Naive Bayes Multinomial");
        algoList.add("SMO");
        outputText = "";
        homepath = "C:\\Users\\Asus\\Documents\\NetBeansProjects\\KategorisasiBeritaWEKA\\src\\java\\tubesai\\weka\\";
    }
//    public File setStopwordsFile(String f)
//    {
//        
//    }
    public void DBtoModel(boolean is_eval) throws Exception {
        File file = new File(homepath + "stopwords.txt");
        swv.setStopwords(file);
        Filter[] mf= {(Filter)nos,(Filter)swv};
        mfilter = new MultiFilter();
        mfilter.setFilters(mf);
        mfilter.setInputFormat(data);
        datafilter = Filter.useFilter(data, mfilter);
        if(algoSelection.equals(algoList.get(0)))
        {
            cls = new NaiveBayesMultinomial();
        }
        else if(algoSelection.equals(algoList.get(1)))
        {
            cls = new SMO();
        }
        cls.buildClassifier(datafilter);
        outputText = "Data classified" + "\n";
        if(is_eval)
        {
            Evaluation eval;
            eval = new Evaluation(datafilter);
            eval.crossValidateModel(cls, data, 10, new Random(1));
            outputText += "Model" + System.lineSeparator() + cls.toString()
                    + eval.toSummaryString("Results", false)+ eval.toClassDetailsString()
                    + eval.toMatrixString();
        }
        
    }
    
    public void Categorize() throws Exception{
        String file = homepath + "test.arff";
        Instances unlabeled = (new DataSource(file)).getDataSet();
        if (unlabeled.classIndex() == -1)
            unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
        Instances unlabeledf = Filter.useFilter(unlabeled, swv);
        Instances labeled = new Instances(unlabeled);
        for (int i = 0; i < unlabeled.numInstances(); i++) {
            double clsLabel = cls.classifyInstance(unlabeledf.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        labeled.deleteAttributeAt(1);
        CSVSaver saver = new CSVSaver();
        File save = new File(homepath + "labeled.csv");
        saver.setInstances(labeled);
        saver.setFile(save);
        saver.setUseRelativePath(false);
        saver.writeBatch();
        outputText = "";
        for (int i = 0; i < unlabeled.numInstances(); i++){
            outputText += i + ". " + labeled.instance(i).stringValue(labeled.classIndex()) + "\n";
        }
    }
}
