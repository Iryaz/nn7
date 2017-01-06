
#include "pugixml.hpp"
#include <string>

#include "nn7feedforward.h"
#include "xmlconfig.h"

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

using std::string;
using std::to_string;
using std::cout;
using namespace pugi;

#define FILE_NAME "save_file_output.xml"

#define INPUTS_NUM 2
#define OUTPUT_NUM 2
#define LAYOUT_NUM 2
#define NEURONS_NUM_FOR_LAYER 6

void createFile()
{
  pugi::xml_document doc;
  pugi::xml_node feedBackConfig = doc.append_child("FeedBackNeuronNetwork");

  pugi::xml_node versionNode = feedBackConfig.append_child("Version");
  versionNode.append_attribute("type") = "double";
  versionNode.append_child(pugi::node_pcdata).set_value("0.9");

  pugi::xml_node inputsNumNode = feedBackConfig.append_child("InputsNum");
  inputsNumNode.append_attribute("type") = "int";
  inputsNumNode.append_child(pugi::node_pcdata).set_value(to_string(INPUTS_NUM).c_str());

  pugi::xml_node outputsNumNode = feedBackConfig.append_child("OutputsNum");
  outputsNumNode.append_attribute("type") = "int";
  outputsNumNode.append_child(pugi::node_pcdata).set_value(to_string(OUTPUT_NUM).c_str());

  pugi::xml_node layersNumNode = feedBackConfig.append_child("LayersNum");
  layersNumNode.append_attribute("type") = "int";
  layersNumNode.append_child(pugi::node_pcdata).set_value(to_string(LAYOUT_NUM).c_str());

  pugi::xml_node neuronNumForLayerNode = feedBackConfig.append_child("NeuronsNumForLayer");
  neuronNumForLayerNode.append_attribute("type") = "int";
  neuronNumForLayerNode.append_child(pugi::node_pcdata).set_value(to_string(NEURONS_NUM_FOR_LAYER).c_str());

  pugi::xml_node weightLstNode = doc.append_child("WeightList");

  for (int l = 0; l < LAYOUT_NUM; l++) {
    string name = "Layout";
    name += std::to_string(l);
    pugi::xml_node layoutNode = weightLstNode.append_child(name.c_str());

    for (int n = 0; n < NEURONS_NUM_FOR_LAYER; n++) {
      string neuronName = "Neuron";
      neuronName += std::to_string(n);
      pugi::xml_node neuronNode = layoutNode.append_child(neuronName.c_str());

      int neuronInputNum = 0;
      if (l == 0)
        neuronInputNum = INPUTS_NUM;
      else
        neuronInputNum = NEURONS_NUM_FOR_LAYER;

      for (int w = 0; w < neuronInputNum; w++) {
        string wname = "Weight";
        wname += std::to_string(w);
        pugi::xml_node wnode = neuronNode.append_child(wname.c_str());
        wnode.append_attribute("type") = "double";
        wnode.append_child(pugi::node_pcdata).set_value("0.0");
      }
    }
  }

  xml_node outLayout = weightLstNode.append_child("OutputLayout");
  for (int n = 0; n < OUTPUT_NUM; n++) {
    string neuronName = "Neuron";
    neuronName += std::to_string(n);
    xml_node neuronNode = outLayout.append_child(neuronName.c_str());

    for (int w = 0; w < NEURONS_NUM_FOR_LAYER; w++) {
      string wname = "Weight";
      wname += std::to_string(w);
      pugi::xml_node wnode = neuronNode.append_child(wname.c_str());
      wnode.append_attribute("type") = "double";
      wnode.append_child(pugi::node_pcdata).set_value("0.0");
    }
  }

  doc.save_file(FILE_NAME);
}

void parseFile()
{
  xml_document doc;
  doc.load_file(FILE_NAME);

  xml_node n = doc.child("FeedBackNeuronNetwork");
  cout << "Version: "<< n.child_value("Version") << "\n";
  cout << "Inputs num: " << n.child_value("InputsNum") << "\n";
  cout << "Output num: " << n.child_value("OutputsNum") << "\n";
  cout << "LayersNum: " << n.child_value("LayersNum") << "\n";
  cout << "NeuronsNumForLayer: " << n.child_value("NeuronsNumForLayer") << "\n";

  xml_node weightNetwork = doc.child("WeightList");

  for (xml_node layout : doc.child("WeightList"))
  {
    //cout << layout.name() << layout.value() << "\n";
    for (xml_node neuron : layout.children())
    {
      for (xml_node weight : neuron.children())
      {
        cout << layout.name() << " "  << neuron.name() << " "
        << weight.name() << " "
        << neuron.child_value(weight.name()) << "\n";
      }
    }
  }
}

TEST_CASE("")
{
  NN7FeedForward<NN7UnipolarSigmoidNeuron> nn(2, 1, 3, 4);
  nn.saveNetwork(FILE_NAME);

  NN7FeedForward<NN7UnipolarSigmoidNeuron> nn1(FILE_NAME);


  print_nn(nn1);
}
