package infrastructure

import (
	"flag"
	"lidar-classification/internal/domain"
	"os"
	"runtime"

	"go.uber.org/zap"
	"gopkg.in/yaml.v3"
)

type YAMLConfigReader struct {
	logger *zap.Logger
}

func NewYAMLConfigReader(logger *zap.Logger) *YAMLConfigReader {
	return &YAMLConfigReader{logger: logger}
}

func (r *YAMLConfigReader) ReadConfig(path string) (*domain.Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config domain.Config
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	// Применяем аргументы командной строки
	r.applyCommandLineFlags(&config)

	// Устанавливаем значения по умолчанию
	r.setDefaults(&config)

	return &config, nil
}

func (r *YAMLConfigReader) applyCommandLineFlags(config *domain.Config) {
	workers := flag.Int("workers", config.Workers, "Number of workers")
	nsamples := flag.Int("nsamples", config.NSamples, "Number of samples")
	n1 := flag.Int("n1", config.N1, "Number of best solutions")
	epsilon := flag.Float64("epsilon", config.Epsilon, "Residual threshold")
	logLevel := flag.String("log-level", config.LogLevel, "Log level")
	method := flag.String("method", config.Method, "Optimization method")

	flag.Parse()

	config.Workers = *workers
	config.NSamples = *nsamples
	config.N1 = *n1
	config.Epsilon = *epsilon
	config.LogLevel = *logLevel
	config.Method = *method
}

func (r *YAMLConfigReader) setDefaults(config *domain.Config) {
	if config.NSamples == 0 {
		config.NSamples = 100
	}
	if config.N1 == 0 {
		config.N1 = 10
	}
	if config.Epsilon == 0 {
		config.Epsilon = 0.1
	}
	if config.Workers == 0 {
		config.Workers = max(1, runtime.NumCPU()-1)
	}
	if config.LogLevel == "" {
		config.LogLevel = "info"
	}
	if config.Method == "" {
		config.Method = "lbfgs"
	}
}
