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

	if min == max {
		min = math.Inf(1)
		max = 0
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
	}
	binWidth := (max - min) / float64(n-1)
	histogram := make([]int, n)
	bins := make([]float64, n)

	for i := range n {
		bins[i] = min + float64(i)*binWidth
	}

	for _, row := range m.Data {
		for _, value := range row {
			if value < min {
				value = min
			} else if value > max {
				value = max
			}
			binIndex := int((value - min) / binWidth)
			histogram[binIndex]++
		}
	}

	return Histogram{
		Bins: bins,
		Vals: histogram,
		Len:  n,
	}, nil
}
