/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubesai.weka;

import com.mysql.jdbc.Driver;
import java.sql.Connection;
import java.io.File;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
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
import weka.core.Attribute;
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
    private final StringToWordVector swv;
    private final WordTokenizer wt;
    private MultiFilter mfilter;
    private Part part1;
    private Part part2;
    private final ArrayList<String> algoList;
    private Classifier cls;
    private String algoSelection;
    private String outputText;
    private boolean is_eval;
    private final String homepath;
    private Instances labeled;
    private ArrayList<ClassifiedData> testlist;
    private final String[] labellist = {"Pendidikan","Politik","Hukum dan Kriminal","Sosial Budaya","Olahraga"
        ,"Teknologi dan Sains","Hiburan","Bisnis dan Ekonomi","Kesehatan","Bencana dan Kecelakaan"};

    public String[] getLabellist() {
        return labellist;
    }
    
    public ArrayList<ClassifiedData> getTestlist() {
        return testlist;
    }

    public void setTestlist(ArrayList<ClassifiedData> testlist) {
        this.testlist = testlist;
    }
    
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
        File file = new File(homepath + "stopwords.txt");
        swv.setStopwords(file);
    }
//    public File setStopwordsFile(String f)
//    {
//        
//    }
    public void DBtoModel() throws Exception {
        
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
            eval.crossValidateModel(cls, datafilter, 10, new Random(1));
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
        labeled = new Instances(unlabeled);
        for (int i = 0; i < unlabeled.numInstances(); i++) {
            double clsLabel = cls.classifyInstance(unlabeledf.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        SavetoCSV();
        AddtoList();
        outputText = "";
//        for (int i = 0; i < unlabeled.numInstances(); i++){
//            outputText += i + ". " + labeled.instance(i).stringValue(labeled.classIndex()) + "\n";
//        }
    }
    
    private void SavetoCSV() throws Exception
    {
        Instances tocsv = new Instances(labeled);
        tocsv.deleteAttributeAt(2);
        tocsv.deleteAttributeAt(1);
        tocsv.deleteAttributeAt(0);
        Attribute at = new Attribute("ID");
        tocsv.insertAttributeAt(at, 0);
        for (int i = 0; i < labeled.numInstances(); i++)
            tocsv.instance(i).setValue(0, i);
        CSVSaver saver = new CSVSaver();
        File save = new File(homepath + "labeled.csv");
        saver.setInstances(tocsv);
        saver.setFile(save);
        saver.setUseRelativePath(true);
        saver.writeBatch();
    }
    
    private void AddtoList() throws Exception
    {
        testlist = new ArrayList<>();
        for (int i = 0; i < labeled.numInstances(); i++)
        {
            ClassifiedData cd = new ClassifiedData();
            
            cd.setJudul(labeled.instance(i).stringValue(0));
            cd.setFull_text(labeled.instance(i).stringValue(1));
            cd.setLabel(labeled.instance(i).stringValue(3));
            cd.setUrl(labeled.instance(i).stringValue(2));
            
            testlist.add(cd);
        }
    }
    
    public void TrainDatabase() throws Exception{
        Class.forName("com.mysql.jdbc.Driver");
        Connection c = DriverManager.getConnection("jdbc:mysql://localhost:3306/news_aggregator","root", "");
        for(ClassifiedData cd : testlist)
        {
            if(cd.Changed())
            {
                String query1 = "INSERT INTO artikel('JUDUL','FULL_TEXT','URL') VALUES('" +
                                cd.judul + "','" + cd.full_text + "','" +
                                cd.url +"');";
                String query2 = "SELECT ID_ARTIKEL FROM artikel WHERE FULL_TEXT='" + cd.full_text +"';";
                Statement stmt = c.createStatement();
                stmt.executeUpdate(query1);
                ResultSet rs = stmt.executeQuery(query2);
                int ret = rs.getInt("ID_ARTIKEL");
                String query3 = "INSERT INTO artikel_kategori_verified VALUES ('" +
                                ret +"','" + cd.label_change+"');";
                stmt.executeUpdate(query3);
            }
        }
        c.close();
    }
    
    public class ClassifiedData
    {
        String judul;
        String full_text;
        String label;
        String label_change;
        String url;

        public String getLabel_change() {
            return label_change;
        }

        public void setLabel_change(String label_change) {
            this.label_change = label_change;
        }

        public ClassifiedData() {
           
        }

        public String getJudul() {
            return judul;
        }

        public void setJudul(String judul) {
            this.judul = judul;
        }

        public String getFull_text() {
            return full_text;
        }

        public void setFull_text(String full_text) {
            this.full_text = full_text;
        }

        public String getLabel() {
            return label;
        }

        public void setLabel(String label) {
            this.label = label;
        }

        public String getUrl() {
            return url;
        }

        public void setUrl(String url) {
            this.url = url;
        }
        
        public boolean Changed()
        {
            return !label.equals(label_change);
        }
        
        
    }
    
    
}
