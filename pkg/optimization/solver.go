package optimization

import (
	"lidar-classification/internal/domain"
	"math"
	"math/rand"
	"sort"

	"github.com/physicist2018/optimization-go/optimization"
	"go.uber.org/zap"
)

const (
	HUGE_VAL = 1000000000.0
)

type MonteCarloOptimizer struct {
	logger *zap.Logger
}

func NewMonteCarloOptimizer(logger *zap.Logger) *MonteCarloOptimizer {
	return &MonteCarloOptimizer{logger: logger}
}

func (o *MonteCarloOptimizer) Solve(data *domain.PointData, config *domain.Config) *domain.Solution {
	var samples []*domain.Solution

	for _ = range config.NSamples {
		sample := o.generateRandomSample(data, config)
		// здесь не обязательно проверять попадание в eps && sample.Residual <= config.Epsilon
		if sample.IsValid {
			samples = append(samples, sample)
		}
	}

	o.logger.Info("number of samples with valid solution", zap.Int("count", len(samples)))
	if len(samples) == 0 {
		return &domain.Solution{IsValid: false}
	}

	// Сортируем по невязке
	sort.Slice(samples, func(i, j int) bool {
		return samples[i].Residual < samples[j].Residual
	})

	// в отсортированном массиме ищем количество решений где невязка не inf
	count := 0
	for _, sample := range samples {
		if !math.IsInf(sample.Residual, 1) {
			count++
		}
	}

	// Берем лучшие N1 решений
	n1 := min(config.N1, count)
	o.logger.Info("average count", zap.Int("count", n1))
	bestSamples := samples[:n1]

	// Усредняем результаты
	avg := o.averageSolutions(bestSamples)

	avg.Difference = CalculateEquations(avg.Fractions.Array(), &config.LR, &config.CV, &avg.Parameters)
	avg.Difference[0] = (1 - avg.Difference[0]) * 100.0
	avg.Difference[1] = (data.DeltaPrime - avg.Difference[1]) / data.DeltaPrime * 100.0
	avg.Difference[2] = (data.Gf - avg.Difference[2]) / data.Gf * 100.0
	avg.Difference[3] = (data.M - avg.Difference[3]) / data.M * 100.0
	o.logger.Info("best", zap.Any("X", avg))
	return avg

}

func (o *MonteCarloOptimizer) generateRandomSample(data *domain.PointData, config *domain.Config) *domain.Solution {
	params := o.generateRandomParameters(config)

	// Решаем систему уравнений
	fractions, residual := o.solveSystem(data, params, config)

	//&& residual <= config.Epsilon &&
	// fractions.D >= 0 && fractions.U >= 0 && fractions.S >= 0 && fractions.W >= 0 &&
	// math.Abs(fractions.D+fractions.U+fractions.S+fractions.W-1.0) <= 0.05
	return &domain.Solution{
		Residual:   residual,
		Fractions:  fractions,
		Parameters: *params,

		IsValid: residual >= 0 && residual < config.Epsilon,
	}
}

func (o *MonteCarloOptimizer) solveSystem(data *domain.PointData, params *domain.Parameters,
	config *domain.Config) (domain.Fractions, float64) {

	var opt optimization.Optimizer
	//nmConf := optimization.DefaultNelderMeadConfig()
	//opt := optimization.NewOptimizedNelderMead(nmConf)

	switch config.GetOptMethod() {
	case domain.MethodNelderMead:
		nmConf := optimization.DefaultNelderMeadConfig()
		nmConf.Tolerance = 1e-5
		opt = optimization.NewOptimizedNelderMead(nmConf)
	case domain.MethodGradientDescent:
		gdConf := optimization.DefaultGradientDescentConfig()
		gdConf.Tolerance = 1e-5
		gdConf.UseRMSprop = true
		opt = optimization.NewAdaptiveGradientDescent(gdConf)
	case domain.MethodSimulatedAnnealing:
		saConf := optimization.DefaultSimulatedAnnealingConfig()
		opt = optimization.NewSimulatedAnnealing(saConf)
	}

	costFunc := NewCostFunction(
		o.logger,
		data,
		params,
		config,
	)

	// Начальное приближение - равные доли
	initial := []float64{0.25, 0.25, 0.25, 0.25}
	result := opt.Optimize(costFunc, initial)

	o.logger.Debug("Optimization result:", zap.Any("result", result))

	fractions := domain.Fractions{
		D: result.X[0],
		U: result.X[1],
		S: result.X[2],
		W: result.X[3],
	}

	return fractions, result.Value
}

func (o *MonteCarloOptimizer) generateRandomParameters(config *domain.Config) *domain.Parameters {
	deltaD := randomInRange(config.DeltaRange.D[0], config.DeltaRange.D[1])
	deltaU := randomInRange(config.DeltaRange.U[0], config.DeltaRange.U[1])
	deltaS := randomInRange(config.DeltaRange.S[0], config.DeltaRange.S[1])
	deltaW := randomInRange(config.DeltaRange.W[0], config.DeltaRange.W[1])
	return &domain.Parameters{
		GfD:         randomInRange(config.GfRange.D[0], config.GfRange.D[1]),
		GfU:         randomInRange(config.GfRange.U[0], config.GfRange.U[1]),
		GfS:         randomInRange(config.GfRange.S[0], config.GfRange.S[1]),
		GfW:         randomInRange(config.GfRange.W[0], config.GfRange.W[1]),
		DeltaDPrime: deltaD / (1 + deltaD),
		DeltaUPrime: deltaU / (1 + deltaU),
		DeltaSPrime: deltaS / (1 + deltaS),
		DeltaWPrime: deltaW / (1 + deltaW),
		MreD:        randomInRange(config.MRange.D[0], config.MRange.D[1]),
		MreU:        randomInRange(config.MRange.U[0], config.MRange.U[1]),
		MreS:        randomInRange(config.MRange.S[0], config.MRange.S[1]),
		MreW:        randomInRange(config.MRange.W[0], config.MRange.W[1]),
	}
}

func (o *MonteCarloOptimizer) averageSolutions(samples []*domain.Solution) *domain.Solution {
	var sumResidual float64
	var sumFractions domain.Fractions
	var sumParams domain.Parameters
	count := len(samples)

	for _, sample := range samples {
		sumResidual += sample.Residual
		sumFractions.D += sample.Fractions.D
		sumFractions.U += sample.Fractions.U
		sumFractions.S += sample.Fractions.S
		sumFractions.W += sample.Fractions.W
		sumParams.GfD += sample.Parameters.GfD
		sumParams.GfU += sample.Parameters.GfU
		sumParams.GfS += sample.Parameters.GfS
		sumParams.GfW += sample.Parameters.GfW
		sumParams.DeltaDPrime += sample.Parameters.DeltaDPrime
		sumParams.DeltaUPrime += sample.Parameters.DeltaUPrime
		sumParams.DeltaSPrime += sample.Parameters.DeltaSPrime
		sumParams.DeltaWPrime += sample.Parameters.DeltaWPrime
		sumParams.MreD += sample.Parameters.MreD
		sumParams.MreU += sample.Parameters.MreU
		sumParams.MreS += sample.Parameters.MreS
		sumParams.MreW += sample.Parameters.MreW
	}

	avgResidual := sumResidual / float64(count)
	avgFractions := domain.Fractions{
		D: sumFractions.D / float64(count),
		U: sumFractions.U / float64(count),
		S: sumFractions.S / float64(count),
		W: sumFractions.W / float64(count),
	}
	avgParams := domain.Parameters{
		GfD:         sumParams.GfD / float64(count),
		GfU:         sumParams.GfU / float64(count),
		GfS:         sumParams.GfS / float64(count),
		GfW:         sumParams.GfW / float64(count),
		DeltaDPrime: sumParams.DeltaDPrime / float64(count),
		DeltaUPrime: sumParams.DeltaUPrime / float64(count),
		DeltaSPrime: sumParams.DeltaSPrime / float64(count),
		DeltaWPrime: sumParams.DeltaWPrime / float64(count),
		MreD:        sumParams.MreD / float64(count),
		MreU:        sumParams.MreU / float64(count),
		MreS:        sumParams.MreS / float64(count),
		MreW:        sumParams.MreW / float64(count),
	}

	return &domain.Solution{
		Residual:   avgResidual,
		Fractions:  avgFractions,
		Parameters: avgParams,
		IsValid:    true,
	}
}

func randomInRange(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}
