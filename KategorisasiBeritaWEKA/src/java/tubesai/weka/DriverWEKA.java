/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubesai.weka;

import com.mysql.jdbc.Driver;
import java.io.File;
import java.sql.DriverManager;
import java.sql.SQLException;
import javax.faces.bean.ManagedBean;
import javax.faces.bean.ApplicationScoped;
import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.DatabaseLoader;
import weka.core.tokenizers.CharacterDelimitedTokenizer;
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
    
    private DatabaseLoader loader;
    private final Instances data;
    private Instances datafilter;
    private Filter[] filters;
    private MultiFilter mfilter;
    private String fiel;
    public DriverWEKA() throws Exception
    {
        data = createInstance();
    }
    public void Drive() throws ClassNotFoundException, Exception {
            System.out.println("Loaded database");
            System.out.println(data.toSummaryString());
            NominalToString nos = new NominalToString();
            nos.setAttributeIndexes("1-2");
            StringToWordVector swv = new StringToWordVector();
            swv.setAttributeIndices("1-2");
            WordTokenizer wt = new WordTokenizer();
            String delimiters = " \r\t\n.,;:\'\"()?!-><#$\\%&*+/@^_=[]{}|`~0123456789";
            wt.setDelimiters(delimiters);
            swv.setTokenizer(wt);
            swv.setLowerCaseTokens(true);
            swv.setOutputWordCounts(true);
            fiel = "C:\\Users\\Asus\\Documents\\stopwords.txt";
            File file = new File(fiel);
            swv.setStopwords(file);
            Filter[] mf= {nos,swv};
            mfilter = new MultiFilter();
            mfilter.setFilters(mf);
            datafilter = Filter.useFilter(data, mfilter);
            System.out.println("Data filtered");
            datafilter.setClass(datafilter.attribute("LABEL"));
            SMO smo = new SMO();
            smo.buildClassifier(data);
            System.out.println("Data classified");
            
            
            
    }
    
    public Instances createInstance() throws ClassNotFoundException, SQLException, Exception
    {
            InstanceQuery loader = new InstanceQuery();
            loader.setUsername("kevhnmay94");
            loader.setPassword("");
            loader.setQuery("SELECT `artikel`.JUDUL,`artikel`.FULL_TEXT,`kategori`.LABEL\n" +
                            "FROM `artikel`\n" +
                            "NATURAL JOIN `artikel_kategori_verified`\n" +
                            "NATURAL JOIN `kategori`");
            Instances structure = loader.retrieveInstances();
            return structure;
    }
}
