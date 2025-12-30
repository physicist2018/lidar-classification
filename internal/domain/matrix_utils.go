package domain

import (
	"errors"
	"math"
)

var ErrInvalidMatrix = errors.New("invalid matrix")

// Hist calculates the histogram of a matrix data within a specified range.
func (m *MatrixData) Hist(min, max float64, n int) (Histogram, error) {
	if m == nil || len(m.Data) == 0 {
		return Histogram{}, ErrInvalidMatrix
	}

	if n <= 0 {
		return Histogram{}, errors.New("number of bins must be positive")
	}

	if min == max {
		min = math.Inf(1)
		max = math.Inf(-1)
		for _, row := range m.Data {
			for _, value := range row {
				if value < min {
					min = value
				}
				if value > max {
					max = value
				}
			}
		}

		// Если все значения одинаковые
		if min == max {
			max = min + 1
		}
	}

	if n == 1 {
		// Обработка особого случая
		histogram := []int{len(m.Data) * len(m.Data[0])}
		bins := []float64{min}
		return Histogram{
			Bins: bins,
			Vals: histogram,
			Len:  1,
		}, nil
	}

	binWidth := (max - min) / float64(n)
	histogram := make([]int, n)
	bins := make([]float64, n)

	for i := range n {
		bins[i] = min + float64(i)*binWidth
	}

	//epsilon := 1e-12 // Для обработки значений на границе

	for _, row := range m.Data {
		for _, value := range row {
			// Корректировка значений в границы
			if value < min {
				value = min
			} else if value > max {
				value = max
			}

			// Вычисление индекса с проверкой границ
			binIndex := int((value - min) / binWidth)

			// Обработка значения равного max (попадает в последний бин)
			if binIndex == n {
				binIndex = n - 1
			} else if binIndex > n || binIndex < 0 {
				// Запасная проверка (не должна срабатывать при корректных данных)
				binIndex = maxInt(0, minInt(n-1, binIndex))
			}

			histogram[binIndex]++
		}
	}

	return Histogram{
		Bins: bins,
		Vals: histogram,
		Len:  n,
	}, nil
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
