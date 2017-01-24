/*
Copyright (C) 2017 Ilya Ryasanov (ryasanov@gmail.com)
This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along
with this program; if not, see <http://www.gnu.org/licenses>.
*/

#ifndef NN7_XMLCONFIG_H_
#define NN7_XMLCONFIG_H_

#include <string>

#include "pugixml.hpp"

#define TO_STR(x) std::to_string(x).c_str()
#define VERSION 0.1

template <class NETWORK> void network2XML(NETWORK& neuralNetwork, const char* fileName)
{
  pugi::xml_document doc;
  pugi::xml_node neuralNetworkSettings = doc.append_child("NeuralNetworkSettings");

  pugi::xml_node versionNode = neuralNetworkSettings.append_child("Version");
  versionNode.append_child(pugi::node_pcdata).set_value(TO_STR(VERSION));

  pugi::xml_node inputsNumNode = neuralNetworkSettings.append_child("InputsNum");
  inputsNumNode.append_attribute("type") = "uint32_t";
  inputsNumNode.append_child(pugi::node_pcdata).set_value(TO_STR(neuralNetwork.getInputsNum()));

  pugi::xml_node outputsNumNode = neuralNetworkSettings.append_child("OutputsNum");
  outputsNumNode.append_attribute("type") = "uint32_t";
  outputsNumNode.append_child(pugi::node_pcdata).set_value(TO_STR(neuralNetwork.getOutputNum()));

  pugi::xml_node layersNumNode = neuralNetworkSettings.append_child("LayersNum");
  layersNumNode.append_attribute("type") = "uint32_t";
  layersNumNode.append_child(pugi::node_pcdata).set_value(TO_STR(neuralNetwork.getHiddenLayersNum()));

  pugi::xml_node neuronsNumNode = neuralNetworkSettings.append_child("NeuronsNumForLayer");
  neuronsNumNode.append_attribute("type") = "uint32_t";
  neuronsNumNode.append_child(pugi::node_pcdata).set_value(TO_STR(neuralNetwork.getNeuronsNumHiddenLayer()));

  pugi::xml_node weightLstNode = doc.append_child("WeightList");

  for (uint32_t l = 0; l < neuralNetwork.getHiddenLayersNum(); l++) {
    std::string name = "Layout";
    name += std::to_string(l);
    pugi::xml_node layoutNode = weightLstNode.append_child(name.c_str());

    for (uint32_t n = 0; n < neuralNetwork.getLayerNeuronsNum(l); n++) {
      std::string neuronName = "Neuron";
      neuronName += std::to_string(n);
      pugi::xml_node neuronNode = layoutNode.append_child(neuronName.c_str());

      uint32_t neuronInputNum = 0;
      if (l == 0)
        neuronInputNum = neuralNetwork.getInputsNum();
      else
        neuronInputNum = neuralNetwork.getLayerNeuronsNum(l);

      pugi::xml_node bnode = neuronNode.append_child("bias");
      bnode.append_attribute("type") = "double";
      double bias = neuralNetwork.getBias(l, n);
      bnode.append_child(pugi::node_pcdata).set_value(TO_STR(bias));

      for (uint32_t w = 0; w < neuronInputNum; w++) {
        std::string wname = "Weight";
        wname += std::to_string(w);
        pugi::xml_node wnode = neuronNode.append_child(wname.c_str());
        wnode.append_attribute("type") = "double";
        double weight = neuralNetwork.getWeight(l, n, w);
        wnode.append_child(pugi::node_pcdata).set_value(TO_STR(weight));
      }
    }
  }

  pugi::xml_node outLayout = weightLstNode.append_child("OutputLayout");
  for (uint32_t n = 0; n < neuralNetwork.getOutputNum(); n++) {
    std::string neuronName = "Neuron";
    neuronName += std::to_string(n);
    pugi::xml_node neuronNode = outLayout.append_child(neuronName.c_str());

    pugi::xml_node bnode = neuronNode.append_child("bias");
    bnode.append_attribute("type") = "double";
    double bias = neuralNetwork.getBias(neuralNetwork.getHiddenLayersNum(), n);
    bnode.append_child(pugi::node_pcdata).set_value(TO_STR(bias));

    for (uint32_t w = 0; w < neuralNetwork.getLayerNeuronsNum(neuralNetwork.getHiddenLayersNum() - 1); w++) {
      std::string wname = "Weight";
      wname += std::to_string(w);
      pugi::xml_node wnode = neuronNode.append_child(wname.c_str());
      wnode.append_attribute("type") = "double";
      uint32_t lastLayout = neuralNetwork.getHiddenLayersNum();
      double weight = neuralNetwork.getWeight(lastLayout, n, w);
      wnode.append_child(pugi::node_pcdata).set_value(TO_STR(weight));
    }
  }

  doc.save_file(fileName);
}

template <class NETWORK> void XML2Network(NETWORK& neuralNetwork, const char* fileName)
{
  pugi::xml_document doc;
  doc.load_file(fileName);

  pugi::xml_node n = doc.child("NeuralNetworkSettings");

  neuralNetwork.setInputsNum(std::stoi(n.child_value("InputsNum")));
  neuralNetwork.setOutputsNum(std::stoi(n.child_value("OutputsNum")));
  neuralNetwork.setHiddenLayersNum(std::stoi(n.child_value("LayersNum")));
  neuralNetwork.setNeuronsNumHiddenLayer(std::stoi(n.child_value("NeuronsNumForLayer")));
  neuralNetwork.initNeuralNetwork();

  pugi::xml_node weightNetwork = doc.child("WeightList");

  uint32_t layoutNum = 0, neuronNum = 0, weightNum = 0;

  for (pugi::xml_node layout : doc.child("WeightList"))
  {
    neuronNum = 0;
    for (pugi::xml_node neuron : layout.children())
    {
      weightNum = 0;
      for (pugi::xml_node weight : neuron.children())
      {
        if (std::string(weight.name()) == "bias") {
          double bias = std::stod(neuron.child_value(weight.name()));
          neuralNetwork.setBias(layoutNum, neuronNum, bias);
        } else {
          double newW = std::stod(neuron.child_value(weight.name()));
          neuralNetwork.setWeight(layoutNum, neuronNum, weightNum , newW);
          weightNum++;
        }
      }
      neuronNum++;
    }

    layoutNum++;
  }
}

#endif
