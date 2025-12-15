package main

import (
	"flag"
	"lidar-classification/internal/app"
	"lidar-classification/internal/domain"
	"lidar-classification/internal/infrastructure"
	"strconv"
	"strings"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
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
	logger = initLogger(config.LogLevel, config.LogFile)

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
		"mre_d":     "mre_d.txt",
		"mre_u":     "mre_u.txt",
		"mre_s":     "mre_s.txt",
		"mre_w":     "mre_w.txt",
	}

	fmtGf := func(val float64) string {
		return strconv.FormatFloat(val, 'f', config.DecimalsGf, 64)
	}

	fmgDefault := func(val float64) string {
		return strconv.FormatFloat(val, 'f', config.DecimalsDefault, 64)
	}

	var fmtStr infrastructure.FmtFunc
	for key, filename := range outputFiles {
		if strings.HasPrefix(key, "GF") {
			fmtStr = fmtGf
		} else {
			fmtStr = fmgDefault
		}

		if err := fileWriter.WriteMatrix(filename, results[key], fmtStr); err != nil {
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

// initLogger initializes the logger with the specified level and log file name.
func initLogger(level string, logfileName ...string) *zap.Logger {
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

	outputPath := make([]string, len(logfileName))
	for i, item := range logfileName {
		outputPath[i] = item
	}

	config.OutputPaths = outputPath
	config.ErrorOutputPaths = outputPath
	config.EncoderConfig.TimeKey = "t"
	config.EncoderConfig.EncodeTime = zapcore.RFC3339TimeEncoder
	config.DisableCaller = false

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
