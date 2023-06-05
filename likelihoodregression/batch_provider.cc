// Compile with g++ batch_provider.cc -o batch_provider `root-config --cflags --glibs` -lRooFit -lRooFitCore -lRooStats

#include "TFile.h"
#include "TTree.h"
#include "RooStats/ModelConfig.h"
#include "RooWorkspace.h"
#include "RooCmdArg.h"
#include "RooAbsPdf.h"
#include "TRandom3.h"
#include "RooRealVar.h"
#include "TString.h" 

class batch_provider{

public: 
  batch_provider(TString file){
    std::cout << "Reading file " << file << std::endl;
    tf = TFile::Open(file);
    w = (RooWorkspace*) tf->Get("w");
    auto mc = dynamic_cast<RooStats::ModelConfig *>(w->genobj("ModelConfig")) ;
    RooAbsPdf &pdf = *mc->GetPdf(); 
    const RooCmdArg &constrainCmdArg = RooFit::Constrain(*mc->GetNuisanceParameters());
    RooAbsData *dobs = w->data("data_obs") ;
    nll = pdf.createNLL(*dobs, constrainCmdArg, RooFit::Extended(pdf.canBeExtended()), RooFit::Offset(true));

    rd  = new TRandom3(0);
    auto params = w->allVars();
    
    dynamic_cast<RooRealVar*>(params.find("cQQ1"))->setRange(-5.0,5.0);
    dynamic_cast<RooRealVar*>(params.find("cQei"))->setRange(-4.0,4.0);
    dynamic_cast<RooRealVar*>(params.find("cQl3i"))->setRange(-5.5,5.5);
    dynamic_cast<RooRealVar*>(params.find("cQlMi"))->setRange(-4.0,4.0);
    dynamic_cast<RooRealVar*>(params.find("cQq11"))->setRange(-0.7,0.7);
    dynamic_cast<RooRealVar*>(params.find("cQq13"))->setRange(-0.35,0.35);
    dynamic_cast<RooRealVar*>(params.find("cQq81"))->setRange(-1.7,1.5);
    dynamic_cast<RooRealVar*>(params.find("cQq83"))->setRange(-0.6,0.6);
    dynamic_cast<RooRealVar*>(params.find("cQt1"))->setRange(-4.0,4.0);
    dynamic_cast<RooRealVar*>(params.find("cQt8"))->setRange(-8.0,8.0);
    dynamic_cast<RooRealVar*>(params.find("cbW"))->setRange(-3.0,3.0);
    dynamic_cast<RooRealVar*>(params.find("cpQ3"))->setRange(-4.0,4.0);
    dynamic_cast<RooRealVar*>(params.find("cpQM"))->setRange(-10.0,17.0);
    dynamic_cast<RooRealVar*>(params.find("cpt"))->setRange(-15.0,15.0);
    dynamic_cast<RooRealVar*>(params.find("cptb"))->setRange(-9.0,9.0);
    dynamic_cast<RooRealVar*>(params.find("ctG"))->setRange(-0.8,0.8);
    dynamic_cast<RooRealVar*>(params.find("ctW"))->setRange(-1.5,1.5);
    dynamic_cast<RooRealVar*>(params.find("ctZ"))->setRange(-2.0,2.0);
    dynamic_cast<RooRealVar*>(params.find("ctei"))->setRange(-4.0,4.0);
    dynamic_cast<RooRealVar*>(params.find("ctlSi"))->setRange(-5.0,5.0);
    dynamic_cast<RooRealVar*>(params.find("ctlTi"))->setRange(-0.9,0.9);
    dynamic_cast<RooRealVar*>(params.find("ctli"))->setRange(-4.0,4.0);
    dynamic_cast<RooRealVar*>(params.find("ctp"))->setRange(-15.0,40.0);
    dynamic_cast<RooRealVar*>(params.find("ctq1"))->setRange(-0.6,0.6);
    dynamic_cast<RooRealVar*>(params.find("ctq8"))->setRange(-1.4,1.4);
    dynamic_cast<RooRealVar*>(params.find("ctt1"))->setRange(-2.6,2.6);

    {
      const RooArgSet* nuisances = w->set("nuisances");
      std::auto_ptr<TIterator> iter(nuisances->createIterator());
      for (RooAbsArg *tmp = (RooAbsArg*) iter->Next(); tmp != 0; tmp = (RooAbsArg*) iter->Next()) {
	RooRealVar *tmpParameter = dynamic_cast<RooRealVar *>(tmp);
	parameters.push_back(tmpParameter->GetName());
	std::cout << "Adding " << tmpParameter->GetName() << std::endl;
      }
    }

    {
      const RooArgSet* POI = w->set("POI");
      std::auto_ptr<TIterator> iter(POI->createIterator());
      for (RooAbsArg *tmp = (RooAbsArg*) iter->Next(); tmp != 0; tmp = (RooAbsArg*) iter->Next()) {
	RooRealVar *tmpParameter = dynamic_cast<RooRealVar *>(tmp);
	if ((strcmp(tmpParameter->GetName(), "r")==0))
	  continue; 
	parameters.push_back(tmpParameter->GetName());
	std::cout << "Adding " << tmpParameter->GetName() << std::endl;
      }
    }


    // const RooArgSet* globalObs = w->set("globalObservables");
    // for (RooAbsArg *tmp = (RooAbsArg*) iter->Next(); tmp != 0; tmp = (RooAbsArg*) iter->Next()) {
    //   RooRealVar *tmpParameter = dynamic_cast<RooRealVar *>(tmp);
    //   if ((strcmp(tmpParameter->GetName(), "r")==0) || (strcmp(tmpParameter->GetName(), "ONE") ==0) || (strcmp(tmpParameter->GetName(),"CMS_th1x") == 0)){
    // 	std::cout << "Ignoring parameter " << tmpParameter->GetName() << std::endl;
    // 	continue;
    //   }
    //   else if (globalObservables->find(tmpParameter->GetName())){
    // 	std::cout << "Ignoring parameter " << tmpParameter->GetName() << " because its a global obs" << std::endl;
    // 	continue;
    //   }
    //   else{
    // 	std::cout << "Adding parameter " << tmpParameter->GetName() << std::endl;
    // 	parameters.push_back(tmpParameter->GetName());
    //   }
    // }


  }

  void get_batch( int nentries, int iChunk){

    TString outName; outName.Form("tree_%d.root",iChunk);
    TFile* outFile = TFile::Open(outName,"recreate");
    TTree* outTree = new TTree("dNLL","dNLL");
    std::map<TString,float> outValues;

    for (auto& parameter : parameters){
      outValues[parameter] = 0.;
      outTree->Branch(parameter, &outValues[parameter]);
    }
    outValues["nll"] = 0.;
    outTree->Branch("nll", &outValues["nll"]);
    
    std::vector<TString> all_pois={"cQQ1",    "cQei",    "cQl3i",    "cQlMi",    "cQq11",    "cQq13",    "cQq81",    "cQq83",    "cQt1",    "cQt8",    "cbW",    "cpQ3",    "cpQM",    "cpt",    "cptb",    "ctG",    "ctW",    "ctZ",    "ctei",    "ctlSi",    "ctlTi",    "ctli",    "ctp",    "ctq1",    "ctq8",    "ctt1"};
    

    auto params = w->allVars();
    for (int i=0; i < nentries; ++i){
      if (i%10 == 0){ std::cout << "Entry " << i << std::endl;}

      for (auto & var : parameters){
	
    	bool isPOI=false;
    	for (auto& poi : all_pois){
    	  if (poi == var) {
    	    isPOI=true;
    	    break;
    	  }
    	}

    	RooRealVar *tmpParameter = dynamic_cast<RooRealVar *>(params.find(var));
    	double value=0;
    	if (isPOI){
    	  value = rd->Uniform( tmpParameter->getMin(), tmpParameter->getMax());
    	}
	else{
	  value = rd->Gaus( 0., 1.);
	}
	outValues[var] = value;
	tmpParameter->setVal( value );
	


      }
      outValues["nll"] = nll->getVal();
      outTree->Fill();
    }
    outTree->Write();
    outFile->Close();
  }

  void set_var_value( TString varname, float value){
    auto params = w->allVars();
    RooRealVar *tmpParameter = dynamic_cast<RooRealVar *>(params.find(varname));
    tmpParameter->setVal(value);
  }

  void scan_all_parameters(){
    auto params = w->allVars();
    std::vector<TString> parameters;
    std::auto_ptr<TIterator> iter(params.createIterator());
    for (RooAbsArg *tmp = (RooAbsArg*) iter->Next(); tmp != 0; tmp = (RooAbsArg*) iter->Next()) {
      RooRealVar *tmpParameter = dynamic_cast<RooRealVar *>(tmp);
      parameters.push_back(tmpParameter->GetName());
    }
    for (auto & var : parameters){
      std::cout << "Scanning over " << var << std::endl;
      RooRealVar *tmpParameter = dynamic_cast<RooRealVar *>(params.find(var));
      for (int j=0; j<60; j++){
	float val = -3.+0.1*j;
	tmpParameter->setVal(val);
	std::cout << val << " " << nll->getVal() << std::endl;
      }

      // back to nominal
      tmpParameter->setVal(0.);
    }
  }

  RooAbsReal* get_nll(){ return nll;}

protected:
  TFile* tf;
  
  RooWorkspace* w;
  RooAbsReal* nll;
  TRandom3* rd;
  std::vector<TString> parameters;
};


int main(int argc, char* argv[])
{
  batch_provider bp("ptz-lj0pt_fullR2_anatest25v01_withAutostats_withSys.root");
  bp.get_batch(10000,atoi(argv[1]));
  return 0;
}
