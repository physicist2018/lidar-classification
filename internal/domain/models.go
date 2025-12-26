package domain

import (
	"errors"
)

// Config представляет конфигурацию приложения
type Config struct {
	LR LRCoeffs `yaml:"LR"`
	CV CVCoeffs `yaml:"CV"`
	//M            MCoeffs    `yaml:"m"`
	MRange          TypeRanges `yaml:"m_range"`
	DeltaRange      TypeRanges `yaml:"delta_range"`
	GfRange         TypeRanges `yaml:"Gf_range"`
	NSamples        int        `yaml:"NSamples"`
	N1              int        `yaml:"N1"`
	Epsilon         float64    `yaml:"epsilon"`
	Workers         int        `yaml:"workers"`
	LogLevel        string     `yaml:"log_level"`
	Method          string     `yaml:"method"`
	LogFile         string     `yaml:"log_file"`
	CostFunction    string     `yaml:"cost_function"`
	DecimalsDefault int        `yaml:"decimals_default"`
	DecimalsGf      int        `yaml:"decimals_gf"`
}

func (c *Config) GetOptMethod() OptimizationMethod {
	switch c.Method {

	case "nelder-mead":
		return MethodNelderMead
	case "gradient":
		return MethodGradientDescent
	case "simann":
		return MethodSimulatedAnnealing

	default:
		return MethodNelderMead
	}
}

type LRCoeffs struct {
	D float64 `yaml:"d"`
	U float64 `yaml:"u"`
	S float64 `yaml:"s"`
	W float64 `yaml:"w"`
}

type CVCoeffs struct {
	D float64 `yaml:"d"`
	U float64 `yaml:"u"`
	S float64 `yaml:"s"`
	W float64 `yaml:"w"`
}

type MCoeffs struct {
	D float64 `yaml:"d"`
	U float64 `yaml:"u"`
	S float64 `yaml:"s"`
	W float64 `yaml:"w"`
}

type TypeRanges struct {
	D []float64 `yaml:"d"`
	U []float64 `yaml:"u"`
	S []float64 `yaml:"s"`
	W []float64 `yaml:"w"`
}

// MatrixData представляет данные матрицы с метками
type MatrixData struct {
	HeightLabels []float64
	TimeLabels   []string
	Data         [][]float64
	Rows, Cols   int
}

// PointData представляет данные для одной точки
type PointData struct {
	I, J       int
	DeltaPrime float64
	Gf         float64
	M          float64
}

// Solution представляет решение для точки
type Solution struct {
	Residual   float64
	Fractions  Fractions
	Parameters Parameters
	IsValid    bool
	Difference []float64
}

type Fractions struct {
	D, U, S, W float64
}

func (f Fractions) Array() []float64 {
	return []float64{f.D, f.U, f.S, f.W}
}

type Parameters struct {
	GfD, GfU, GfS, GfW                                 float64
	DeltaDPrime, DeltaUPrime, DeltaSPrime, DeltaWPrime float64
	MreD, MreU, MreS, MreW                             float64
}

type ClassifyResults map[string]*MatrixData

type Histogram struct {
	Bins []float64
	Vals []int
	Len  int
}

// OptimizationMethod представляет метод оптимизации
type OptimizationMethod int

const (
	MethodNelderMead OptimizationMethod = iota
	MethodGradientDescent
	MethodSimulatedAnnealing
)

var (
	ErrInvalidFileFormat = errors.New("invalid file format")
)
