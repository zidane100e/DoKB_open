import kr.co.shineware.nlp.komoran.modeler.builder.ModelBuilder;

import java.io.File;

public class ModelBuildTest{
    public static void main(String[] args){
	modelSave();
    }

    private static void modelSave(){
	ModelBuilder builder = new ModelBuilder();
	builder.setExternalDic("../corpus_build/user.dic");
	builder.buildPath("../corpus_build");
	builder.save("models");
    }


}

