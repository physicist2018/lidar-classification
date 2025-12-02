package optimization

import (
	"lidar-classification/internal/domain"
	"math"
	"math/rand"
	"sort"

	"go.uber.org/zap"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/optimize"
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

	Func := func(x []float64) float64 {
		// x = [n_d, n_u, n_s, n_w]
		if len(x) != 4 {
			return math.Inf(1)
		}

		nd, nu, ns, nw := x[0], x[1], x[2], x[3]

		// Ограничения неотрицательности
		if nd < 0 || nu < 0 || ns < 0 || nw < 0 {
			return math.Inf(1)
		}

		// Вычисляем V_k
		vd := nd * config.LR.D * config.CV.D
		vu := nu * config.LR.U * config.CV.U
		vs := ns * config.LR.S * config.CV.S
		vw := nw * config.LR.W * config.CV.W
		vTotal := vd + vu + vs + vw

		// Уравнения
		eq1 := nd + nu + ns + nw - 1.0 // Сумма долей = 1
		eq2 := deltaPrimeD*nd + deltaPrimeU*nu + deltaPrimeS*ns + deltaPrimeW*nw - data.DeltaPrime
		eq3 := params.GfD*nd + params.GfU*nu + params.GfS*ns + params.GfW*nw - data.Gf

		var eq4 float64
		if vTotal > 0 {
			mCalc := (config.M.D*vd + config.M.U*vu + config.M.S*vs + config.M.W*vw) / vTotal
			eq4 = mCalc - data.M
		} else {
			eq4 = math.Inf(1)
		}

		// Невязка как норма Фробениуса относительных отклонений
		residual := math.Sqrt(
			math.Pow(eq1/1.0, 2) +
				math.Pow(eq2/data.DeltaPrime, 2) +
				math.Pow(eq3/data.Gf, 2) +
				math.Pow(eq4/data.M, 2))
		o.logger.Debug("Residual", zap.Float64("residual", residual))
		return residual
	}
	// Grad вычисляет градиент функции в точке x и сохраняет результат в grad
	Grad := func(grad, x []float64) {
		// Проверка размерности
		if len(grad) != 4 || len(x) != 4 {
			for i := range grad {
				grad[i] = math.NaN()
			}
			return
		}

		// Конфигурация для численного дифференцирования
		settings := &fd.Settings{
			Formula: fd.Central, // Центральные разности для большей точности
			Step:    1e-6,       // Шаг дифференцирования
		}

		// Вычисляем градиент численно
		fd.Gradient(grad, Func, x, settings)
	}
	problem := optimize.Problem{
		Func: Func,
		Grad: Grad,
	}

	// Начальное приближение - равные доли
	initial := []float64{0.25, 0.25, 0.25, 0.25}

	result, err := optimize.Minimize(problem, initial, nil, config.GetMethod().ToGonumMethod())
	if err != nil {
		return domain.Fractions{}, math.Inf(1)
	}

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

	return fractions, result.F
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
