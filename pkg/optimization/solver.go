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

	for k := 0; k < config.NSamples; k++ {
		sample := o.generateRandomSample(data, config)
		if sample.IsValid && sample.Residual <= config.Epsilon {
			samples = append(samples, sample)
		}
	}

	if len(samples) == 0 {
		return &domain.Solution{IsValid: false}
	}

	// Сортируем по невязке
	sort.Slice(samples, func(i, j int) bool {
		return samples[i].Residual < samples[j].Residual
	})

	// Берем лучшие N1 решений
	n1 := min(config.N1, len(samples))
	bestSamples := samples[:n1]

	// Усредняем результаты
	return o.averageSolutions(bestSamples)
}

func (o *MonteCarloOptimizer) generateRandomSample(data *domain.PointData, config *domain.Config) *domain.Solution {
	params := o.generateRandomParameters(config)

	// Преобразуем delta в delta'
	deltaPrimeD := params.DeltaD / (1 + params.DeltaD)
	deltaPrimeU := params.DeltaU / (1 + params.DeltaU)
	deltaPrimeS := params.DeltaS / (1 + params.DeltaS)
	deltaPrimeW := params.DeltaW / (1 + params.DeltaW)

	// Решаем систему уравнений
	fractions, residual := o.solveSystem(data, params, deltaPrimeD, deltaPrimeU, deltaPrimeS, deltaPrimeW, config)

	return &domain.Solution{
		Residual:   residual,
		Fractions:  fractions,
		Parameters: *params,
		IsValid: residual >= 0 && residual <= config.Epsilon &&
			fractions.D >= 0 && fractions.U >= 0 && fractions.S >= 0 && fractions.W >= 0 &&
			math.Abs(fractions.D+fractions.U+fractions.S+fractions.W-1.0) <= 0.05,
	}
}

func (o *MonteCarloOptimizer) solveSystem(data *domain.PointData, params *domain.Parameters,
	deltaPrimeD, deltaPrimeU, deltaPrimeS, deltaPrimeW float64,
	config *domain.Config) (domain.Fractions, float64) {

	nmConf := optimization.DefaultNelderMeadConfig()
	opt := optimization.NewOptimizedNelderMead(nmConf)

	costFunc := NewCostFunction(
		o.logger,
		data,
		params,
		deltaPrimeD, deltaPrimeU, deltaPrimeS, deltaPrimeW,
		config,
	)

	// Начальное приближение - равные доли
	initial := []float64{0.25, 0.25, 0.25, 0.25}
	result := opt.Optimize(costFunc, initial)

	// result, err := optimize.Minimize(problem, initial, &optimize.Settings{
	// 	Converger: &optimize.FunctionConverge{
	// 		Absolute:   1e-5,
	// 		Iterations: 100,
	// 	},
	// }, config.GetMethod().ToGonumMethod())
	// if err != nil {
	// 	return domain.Fractions{}, math.Inf(1)
	// }

	fractions := domain.Fractions{
		D: math.Max(0, result.X[0]),
		U: math.Max(0, result.X[1]),
		S: math.Max(0, result.X[2]),
		W: math.Max(0, result.X[3]),
	}

	// Нормализуем доли
	sum := fractions.D + fractions.U + fractions.S + fractions.W
	if sum > 0 {
		fractions.D /= sum
		fractions.U /= sum
		fractions.S /= sum
		fractions.W /= sum
	}

	return fractions, result.Value
}

func (o *MonteCarloOptimizer) generateRandomParameters(config *domain.Config) *domain.Parameters {
	return &domain.Parameters{
		GfD:    randomInRange(config.GfRange.D[0], config.GfRange.D[1]),
		GfU:    randomInRange(config.GfRange.U[0], config.GfRange.U[1]),
		GfS:    randomInRange(config.GfRange.S[0], config.GfRange.S[1]),
		GfW:    randomInRange(config.GfRange.W[0], config.GfRange.W[1]),
		DeltaD: randomInRange(config.DeltaRange.D[0], config.DeltaRange.D[1]),
		DeltaU: randomInRange(config.DeltaRange.U[0], config.DeltaRange.U[1]),
		DeltaS: randomInRange(config.DeltaRange.S[0], config.DeltaRange.S[1]),
		DeltaW: randomInRange(config.DeltaRange.W[0], config.DeltaRange.W[1]),
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
		sumParams.DeltaD += sample.Parameters.DeltaD
		sumParams.DeltaU += sample.Parameters.DeltaU
		sumParams.DeltaS += sample.Parameters.DeltaS
		sumParams.DeltaW += sample.Parameters.DeltaW
	}

	avgResidual := sumResidual / float64(count)
	avgFractions := domain.Fractions{
		D: sumFractions.D / float64(count),
		U: sumFractions.U / float64(count),
		S: sumFractions.S / float64(count),
		W: sumFractions.W / float64(count),
	}
	avgParams := domain.Parameters{
		GfD:    sumParams.GfD / float64(count),
		GfU:    sumParams.GfU / float64(count),
		GfS:    sumParams.GfS / float64(count),
		GfW:    sumParams.GfW / float64(count),
		DeltaD: sumParams.DeltaD / float64(count),
		DeltaU: sumParams.DeltaU / float64(count),
		DeltaS: sumParams.DeltaS / float64(count),
		DeltaW: sumParams.DeltaW / float64(count),
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
