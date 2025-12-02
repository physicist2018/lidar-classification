package domain

import (
	"errors"
	"gonum.org/v1/gonum/optimize"
)

// Config представляет конфигурацию приложения
type Config struct {
	LR         LRCoeffs   `yaml:"LR"`
	CV         CVCoeffs   `yaml:"CV"`
	M          MCoeffs    `yaml:"m"`
	DeltaRange TypeRanges `yaml:"delta_range"`
	GfRange    TypeRanges `yaml:"Gf_range"`
	NSamples   int        `yaml:"NSamples"`
	N1         int        `yaml:"N1"`
	Epsilon    float64    `yaml:"epsilon"`
	Workers    int        `yaml:"workers"`
	LogLevel   string     `yaml:"log_level"`
	Method     string     `yaml:"method"`
}

func (c *Config) GetMethod() OptimizationMethod {
	switch c.Method {
	case "newton":
		return MethodNewton
	case "lbfgs":
		return MethodLBFGS
	case "nelder-mead":
		return MethodNelderMead
	case "gradient":
		return MethodGradient
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
}

type Fractions struct {
	D, U, S, W float64
}

type Parameters struct {
	GfD, GfU, GfS, GfW             float64
	DeltaD, DeltaU, DeltaS, DeltaW float64
}

// OptimizationMethod представляет метод оптимизации
type OptimizationMethod int

const (
	MethodNewton OptimizationMethod = iota
	MethodLBFGS
	MethodNelderMead
	MethodGradient
)

func (m OptimizationMethod) ToGonumMethod() optimize.Method {
	switch m {
	case MethodNewton:
		return &optimize.Newton{}
	case MethodLBFGS:
		return &optimize.LBFGS{}
	case MethodNelderMead:
		return &optimize.NelderMead{}
	case MethodGradient:
		return &optimize.GradientDescent{}
	default:
		return &optimize.LBFGS{}
	}
}

var (
	ErrInvalidFileFormat = errors.New("invalid file format")
)
