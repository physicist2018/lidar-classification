package infrastructure

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"go.uber.org/zap"

	"lidar-classification/internal/domain"
)

type FmtFunc func(float64) string

type TXTFileWriter struct {
	logger *zap.Logger
}

func NewTXTFileWriter(logger *zap.Logger) *TXTFileWriter {
	return &TXTFileWriter{logger: logger}
}

func (w *TXTFileWriter) WriteMatrix(filename string, data *domain.MatrixData, formatter FmtFunc) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	// Записываем метки времени
	timeLabels := strings.Join(data.TimeLabels, "\t")
	fmt.Fprintf(writer, "Alt/Time\t%s\n", timeLabels)

	// Записываем данные с метками высот
	for i, row := range data.Data {
		heightLabel := strconv.FormatFloat(data.HeightLabels[i], 'f', 2, 64)
		var rowStr []string
		for _, val := range row {
			rowStr = append(rowStr, formatter(val))
		}
		fmt.Fprintf(writer, "%s\t%s\n", heightLabel, strings.Join(rowStr, "\t"))
	}

	return nil
}

func (w *TXTFileWriter) WriteHistogram(filename string, hist *domain.Histogram) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	// Записываем метки времени
	timeLabels := strings.Join([]string{"X", "Y"}, "\t")
	fmt.Fprintf(writer, "%s\n", timeLabels)

	// Записываем данные с метками высот
	for i := range hist.Len {
		fmt.Fprintf(writer, "%.2e\t%10d\n", hist.Bins[i], hist.Vals[i])
	}

	return nil
}
