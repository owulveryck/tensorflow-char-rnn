package main

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"

	"github.com/kelseyhightower/envconfig"
)

const (
	resultJSON = "result.json"
	vocabJSON  = "vocab.json"
)

var (
	startText string
	length    int
)

type specifications struct {
	// InitDir points to the directory that contains output informations
	InitDir string `envconfig:"init_dir" required:"true"`
	// Temperature for sampling from softmax:
	// higher temperature, more random
	// lower temperature, more greedy.
	Temperature float32 `envconfig:"temperature" default:"1.0" required:"true"`
	// Seed for sampling to replicate results (0<=seed<=4294967295)
	Seed int64 `required:"true" default:"-1"`
	// BestModel points to the path to the model file (like output/best_model/model-40)
	BestModel    string  `envconfig:"model_path" json:"best_model"`
	BestValidPpl float64 `json:"best_valid_ppl"`
	Encoding     string  `envconfig:"encoding" default:"utf-8" json:"encoding"`
	LatestModel  string  `json:"latest_model"`
	Params       struct {
		BatchSize     int     `json:"batch_size"`
		Dropout       float64 `json:"dropout"`
		EmbeddingSize int     `json:"embedding_size"`
		HiddenSize    int     `json:"hidden_size"`
		InputDropout  float64 `json:"input_dropout"`
		LearningRate  float64 `json:"learning_rate"`
		MaxGradNorm   float64 `json:"max_grad_norm"`
		Model         string  `json:"model"`
		NumLayers     int     `json:"num_layers"`
		NumUnrollings int     `json:"num_unrollings"`
		VocabSize     int     `json:"vocab_size"`
	} `json:"params"`
	TestPpl   float64 `json:"test_ppl"`
	VocabFile string  `json:"vocab_file"`
}

func main() {
	var s specifications
	err := envconfig.Process("RNN", &s)
	if err != nil {
		log.Fatal(err.Error())
	}
	resultFile := filepath.Join(s.InitDir, resultJSON)
	f, err := os.Open(resultFile)
	if err != nil {
		log.Fatal("Cannot read result.json", err)
	}
	dec := json.NewDecoder(f)
	if err := dec.Decode(&s); err != nil {
		f.Close()
		log.Fatal("result not valid", err)
	}
	f.Close()
	vocabFile := filepath.Join(s.InitDir, vocabJSON)
	var vocabIndexDict map[string]int
	f, err = os.Open(vocabFile)
	if err != nil {
		log.Fatal("Cannot read result.json", err)
	}
	dec = json.NewDecoder(f)
	if err := dec.Decode(&vocabIndexDict); err != nil {
		f.Close()
		log.Fatal("result not valid", err)
	}
	f.Close()
	vocabSize := len(vocabIndexDict)
	indexVocabDict := make(map[int]string, vocabSize)
	for k, v := range vocabIndexDict {
		indexVocabDict[v] = k
	}

}
