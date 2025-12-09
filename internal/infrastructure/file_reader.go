package infrastructure

import (
	"bufio"
	"lidar-classification/internal/domain"
	"math"
	"os"
	"strconv"
	"strings"

	"go.uber.org/zap"
)

type TXTFileReader struct {
	logger *zap.Logger
}

func NewTXTFileReader(logger *zap.Logger) *TXTFileReader {
	return &TXTFileReader{logger: logger}
}

func (r *TXTFileReader) ReadMatrix(filename string) (*domain.MatrixData, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	if len(lines) < 2 {
		return nil, domain.ErrInvalidFileFormat
	}

	// Парсим первую строку (метки времени)
	timeLabels := strings.Fields(lines[0])[1:] // Пропускаем первый элемент

	var heightLabels []float64
	var data [][]float64

	// Парсим остальные строки
	for i := 1; i < len(lines); i++ {
		fields := strings.Fields(lines[i])
		if len(fields) < 2 {
			continue
		}

		// Первый столбец - метка высоты
		height, err := strconv.ParseFloat(fields[0], 64)
		if err != nil {
			return nil, err
		}
		heightLabels = append(heightLabels, height)

		// Остальные столбцы - данные
		var row []float64
		for j := 1; j < len(fields); j++ {
			value, err := strconv.ParseFloat(fields[j], 64)
			if err != nil {
				return nil, err
			}
			if value < 0 {
				r.logger.Warn("Negative value found, replaced with NaN", zap.Float64("value", value))
				value = math.NaN()
			}
			row = append(row, value)
		}
		data = append(data, row)
	}

	return &domain.MatrixData{
		HeightLabels: heightLabels,
		TimeLabels:   timeLabels,
		Data:         data,
		Rows:         len(data),
		Cols:         len(data[0]),
	}, nil
}
