package main

import (
	"flag"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"lidar-classification/internal/app"
	"lidar-classification/internal/domain"
	"lidar-classification/internal/infrastructure"
)

func main() {
	configPath := flag.String("config", "config.yaml", "Path to config file")
	flag.Parse()

	// Инициализация логгера
	logger := initLogger("info")
	defer logger.Sync()

	// Чтение конфигурации
	configReader := infrastructure.NewYAMLConfigReader(logger)
	config, err := configReader.ReadConfig(*configPath)
	if err != nil {
		logger.Fatal("Failed to read config", zap.Error(err))
	}

	// Обновляем уровень логирования
	logger = initLogger(config.LogLevel)

	// Инициализация компонентов
	fileReader := infrastructure.NewTXTFileReader(logger)
	fileWriter := infrastructure.NewTXTFileWriter(logger)
	classifier := app.NewAerosolClassifier(logger, config)

	// Чтение входных данных
	depData, err := fileReader.ReadMatrix("Dep.txt")
	if err != nil {
		logger.Fatal("Failed to read Dep.txt", zap.Error(err))
	}

	flData, err := fileReader.ReadMatrix("FL_cap.txt")
	if err != nil {
		logger.Fatal("Failed to read FL_cap.txt", zap.Error(err))
	}

	mreData, err := fileReader.ReadMatrix("mre.txt")
	if err != nil {
		logger.Fatal("Failed to read mre.txt", zap.Error(err))
	}

	// Проверка совместимости размеров
	if !validateMatrixSizes(depData, flData, mreData) {
		logger.Fatal("Input matrices have incompatible sizes")
	}

	logger.Info("Starting aerosol classification",
		zap.Int("rows", depData.Rows),
		zap.Int("cols", depData.Cols),
		zap.Int("workers", config.Workers))

	// Обработка данных
	results := classifier.ProcessMatrices(depData, flData, mreData)

	// Сохранение меток
	for _, result := range results {
		result.HeightLabels = depData.HeightLabels
		result.TimeLabels = depData.TimeLabels
	}

	// Запись результатов
	outputFiles := map[string]string{
		"residuals": "residuals.txt",
		"n_d":       "n_d.txt",
		"n_u":       "n_u.txt",
		"n_s":       "n_s.txt",
		"n_w":       "n_w.txt",
		"GF_d":      "GF_d.txt",
		"GF_u":      "GF_u.txt",
		"GF_s":      "GF_s.txt",
		"GF_w":      "GF_w.txt",
		"delta_d":   "delta_d.txt",
		"delta_u":   "delta_u.txt",
		"delta_s":   "delta_s.txt",
		"delta_w":   "delta_w.txt",
	}

	for key, filename := range outputFiles {
		if err := fileWriter.WriteMatrix(filename, results[key]); err != nil {
			logger.Error("Failed to write result",
				zap.String("file", filename),
				zap.Error(err))
		} else {
			logger.Info("Successfully written result",
				zap.String("file", filename))
		}
	}

	logger.Info("Aerosol classification completed successfully")
}

func initLogger(level string) *zap.Logger {
	config := zap.NewProductionConfig()

	switch level {
	case "debug":
		config.Level = zap.NewAtomicLevelAt(zap.DebugLevel)
	case "warn":
		config.Level = zap.NewAtomicLevelAt(zap.WarnLevel)
	case "error":
		config.Level = zap.NewAtomicLevelAt(zap.ErrorLevel)
	default:
		config.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
	}

	config.EncoderConfig.TimeKey = "time"
	config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder

	logger, _ := config.Build()
	return logger
}

func validateMatrixSizes(matrices ...*domain.MatrixData) bool {
	if len(matrices) < 2 {
		return true
	}

	rows, cols := matrices[0].Rows, matrices[0].Cols
	for _, matrix := range matrices[1:] {
		if matrix.Rows != rows || matrix.Cols != cols {
			return false
		}
	}
	return true
}
