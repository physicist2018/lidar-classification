package optimization

import (
	"lidar-classification/internal/domain"
	"math"

	"go.uber.org/zap"
)

type CostFunction struct {
	logger                                             *zap.Logger
	data                                               *domain.PointData
	params                                             *domain.Parameters
	deltaPrimeD, deltaPrimeU, deltaPrimeS, deltaPrimeW float64
	conf                                               *domain.Config
}

func NewCostFunction(logger *zap.Logger, data *domain.PointData, params *domain.Parameters, deltaPrimeD, deltaPrimeU, deltaPrimeS, deltaPrimeW float64, conf *domain.Config) *CostFunction {
	return &CostFunction{
		logger:      logger,
		data:        data,
		params:      params,
		deltaPrimeD: deltaPrimeD,
		deltaPrimeU: deltaPrimeU,
		deltaPrimeS: deltaPrimeS,
		deltaPrimeW: deltaPrimeW,
		conf:        conf,
	}
}

// func (c *CostFunction) Value(x []float64) float64 {
// 	// x = [n_d, n_u, n_s, n_w]
// 	if len(x) != 4 {
// 		return math.Inf(1)
// 	}

// 	nd, nu, ns, nw := x[0], x[1], x[2], x[3]
// 	c.logger.Debug("X", zap.Float64("nd", nd), zap.Float64("nu", nu), zap.Float64("ns", ns), zap.Float64("nw", nw))
// 	// Ограничения неотрицательности
// 	if nd < 0 || nu < 0 || ns < 0 || nw < 0 {
// 		return HUGE_VAL
// 	}

// 	// Вычисляем V_k
// 	vd := nd * c.conf.LR.D * c.conf.CV.D
// 	vu := nu * c.conf.LR.U * c.conf.CV.U
// 	vs := ns * c.conf.LR.S * c.conf.CV.S
// 	vw := nw * c.conf.LR.W * c.conf.CV.W
// 	vTotal := vd + vu + vs + vw

// 	// Уравнения
// 	eq1 := nd + nu + ns + nw - 1.0 // Сумма долей = 1
// 	eq2 := c.deltaPrimeD*nd + c.deltaPrimeU*nu + c.deltaPrimeS*ns + c.deltaPrimeW*nw - c.data.DeltaPrime
// 	eq3 := c.params.GfD*nd + c.params.GfU*nu + c.params.GfS*ns + c.params.GfW*nw - c.data.Gf

// 	var eq4 float64
// 	if vTotal > 0 {
// 		mCalc := (c.conf.M.D*vd + c.conf.M.U*vu + c.conf.M.S*vs + c.conf.M.W*vw) / vTotal
// 		eq4 = mCalc - c.data.M
// 	} else {
// 		eq4 = HUGE_VAL
// 	}

// 	// Невязка как норма Фробениуса относительных отклонений
// 	eps1 := math.Pow(eq1/1.0, 2)
// 	eps2 := math.Pow(eq2/c.data.DeltaPrime, 2)
// 	eps3 := math.Pow(eq3/c.data.Gf, 2)
// 	eps4 := math.Pow(eq4/c.data.M, 2)

// 	residual := math.Sqrt(
// 		eps1 + eps2 + eps3 + eps4)
// 	if eps1 > 0.01 {
// 		residual += HUGE_VAL
// 	}

// 	c.logger.Debug("Residuals:", zap.Float64("eps_n", eps1),
// 		zap.Float64("eps_delta", eps2),
// 		zap.Float64("eps_gf", eps3),
// 		zap.Float64("eps_m", eps4),
// 		zap.Float64("residual", residual))
// 	return residual
// }

func (c *CostFunction) Value(x []float64) float64 {
	// Проверка размерности
	if len(x) != 4 {
		return 1e10 // Большое, но конечное значение
	}

	nd, nu, ns, nw := x[0], x[1], x[2], x[3]

	c.logger.Debug("Input",
		zap.Float64("nd", nd),
		zap.Float64("nu", nu),
		zap.Float64("ns", ns),
		zap.Float64("nw", nw))

	// Плавное наказание за отрицательность (квадратичное)
	var penalty float64
	// for _, val := range []float64{nd, nu, ns, nw} {
	// 	if val < 0 {
	// 		penalty += 1e6 * val * val // Плавный штраф
	// 	}
	// }

	// Вычисление объёмов
	vd := nd * c.conf.LR.D * c.conf.CV.D
	vu := nu * c.conf.LR.U * c.conf.CV.U
	vs := ns * c.conf.LR.S * c.conf.CV.S
	vw := nw * c.conf.LR.W * c.conf.CV.W
	vTotal := vd + vu + vs + vw

	// Уравнения (остатки)
	eq1 := nd + nu + ns + nw - 1.0 // Сумма долей = 1
	eq2 := c.deltaPrimeD*nd + c.deltaPrimeU*nu +
		c.deltaPrimeS*ns + c.deltaPrimeW*nw - c.data.DeltaPrime // DeltaPrime
	eq3 := c.params.GfD*nd + c.params.GfU*nu +
		c.params.GfS*ns + c.params.GfW*nw - c.data.Gf // Gf

	var eq4 float64
	if vTotal > 1e-8 { // Регуляризация при малом vTotal
		mCalc := (c.conf.M.D*vd + c.conf.M.U*vu +
			c.conf.M.S*vs + c.conf.M.W*vw) / vTotal
		eq4 = mCalc - c.data.M
	} else {
		eq4 = 0 // Нейтральное значение при vTotal ≈ 0
	}

	// Адаптивные веса для нормализации остатков
	weight1 := 1.0
	weight2 := 1.0 / math.Max(1e-8, math.Abs(c.data.DeltaPrime))
	weight3 := 1.0 / math.Max(1e-8, math.Abs(c.data.Gf))
	weight4 := 1.0 / math.Max(1e-8, math.Abs(c.data.M))

	// Квадраты нормированных остатков
	eps1 := weight1 * eq1
	eps2 := weight2 * eq2
	eps3 := weight3 * eq3
	eps4 := weight4 * eq4

	residual := math.Sqrt(
		eps1*eps1 + eps2*eps2 + eps3*eps3 + eps4*eps4,
	)

	// Дополнительное мягкое наказание за большое отклонение суммы долей
	if math.Abs(eq1) > 0.01 {
		residual += 1e4 * math.Pow(eq1, 2)
	}

	// Итоговый результат: невязка + штраф
	total := residual + penalty

	c.logger.Debug("Residuals",
		zap.Float64("eps_n", eps1),
		zap.Float64("eps_delta", eps2),
		zap.Float64("eps_gf", eps3),
		zap.Float64("eps_m", eps4),
		zap.Float64("residual", residual),
		zap.Float64("penalty", penalty),
		zap.Float64("total", total))

	return total
}

func (c *CostFunction) Value1(x []float64) float64 {
	// x = [n_d, n_u, n_s, n_w]
	if len(x) != 4 {
		return math.Inf(1)
	}

	nd, nu, ns, nw := x[0], x[1], x[2], x[3]
	c.logger.Debug("X", zap.Float64("nd", nd), zap.Float64("nu", nu), zap.Float64("ns", ns), zap.Float64("nw", nw))

	// Гладкие штрафы за неотрицательность (экспоненциальный штраф)
	negPenalty := 0.0
	if nd < 0 {
		negPenalty += math.Exp(-1000.0 * nd)
	}
	if nu < 0 {
		negPenalty += math.Exp(-1000.0 * nu)
	}
	if ns < 0 {
		negPenalty += math.Exp(-1000.0 * ns)
	}
	if nw < 0 {
		negPenalty += math.Exp(-1000.0 * nw)
	}

	// Вычисляем V_k
	vd := nd * c.conf.LR.D * c.conf.CV.D
	vu := nu * c.conf.LR.U * c.conf.CV.U
	vs := ns * c.conf.LR.S * c.conf.CV.S
	vw := nw * c.conf.LR.W * c.conf.CV.W
	vTotal := vd + vu + vs + vw

	// Уравнения
	eq1 := nd + nu + ns + nw - 1.0 // Сумма долей = 1
	eq2 := c.deltaPrimeD*nd + c.deltaPrimeU*nu + c.deltaPrimeS*ns + c.deltaPrimeW*nw - c.data.DeltaPrime
	eq3 := c.params.GfD*nd + c.params.GfU*nu + c.params.GfS*ns + c.params.GfW*nw - c.data.Gf

	var eq4 float64
	if vTotal > 1e-12 { // Небольшой порог для избежания деления на ноль
		mCalc := (c.conf.M.D*vd + c.conf.M.U*vu + c.conf.M.S*vs + c.conf.M.W*vw) / vTotal
		eq4 = mCalc - c.data.M
	} else {
		// Гладкий штраф при нулевом vTotal
		vTarget := c.conf.LR.D*c.conf.CV.D*0.25 + // Примерное ожидаемое значение
			c.conf.LR.U*c.conf.CV.U*0.25 +
			c.conf.LR.S*c.conf.CV.S*0.25 +
			c.conf.LR.W*c.conf.CV.W*0.25
		eq4 = HUGE_VAL * (1.0 - math.Exp(-vTotal/vTarget*10.0))
	}

	// Невязка как норма Фробениуса относительных отклонений
	eps1 := math.Pow(eq1, 2) // Абсолютная ошибка (вместо относительной)
	eps2 := math.Pow(eq2/(math.Abs(c.data.DeltaPrime)+1e-12), 2)
	eps3 := math.Pow(eq3/(math.Abs(c.data.Gf)+1e-12), 2)
	eps4 := math.Pow(eq4/(math.Abs(c.data.M)+1e-12), 2)

	// Гладкий штраф за нарушение нормировки вместо разрыва
	sumPenalty := 0.0
	if math.Abs(eq1) > 0.01 {
		sumPenalty += 1000.0 * math.Pow(math.Max(0, math.Abs(eq1)-0.01), 2)
	}

	residual := math.Sqrt(eps1+eps2+eps3+eps4) +
		10.0*negPenalty + // Штраф за отрицательность
		sumPenalty // Штраф за нарушение нормировки

	c.logger.Debug("Residuals:", zap.Float64("eps_n", eps1),
		zap.Float64("eps_delta", eps2),
		zap.Float64("eps_gf", eps3),
		zap.Float64("eps_m", eps4),
		zap.Float64("negPenalty", negPenalty),
		zap.Float64("sumPenalty", sumPenalty),
		zap.Float64("residual", residual))
	return residual
}

// Gradient вычисляет градиент функции в точке x и сохраняет результат в gradient
func (c *CostFunction) Gradient(x []float64) []float64 {
	n := len(x)
	gradient := make([]float64, n)
	h := 1e-8
	fx := c.Value(x)
	for i := 0; i < n; i++ {
		x[i] += h
		fxh := c.Value(x)
		gradient[i] = (fxh - fx) / h
		x[i] -= h
	}
	c.logger.Debug("Gradient:", zap.Float64s("gradient", gradient))
	return gradient
}

// Grad вычисляет градиент функции в точке x и сохраняет результат в grad
// Grad := func(grad, x []float64) {
// 	// Проверка размерности
// 	if len(grad) != 4 || len(x) != 4 {
// 		for i := range grad {
// 			grad[i] = math.NaN()
// 		}
// 		return
// 	}

// 	// Конфигурация для численного дифференцирования
// 	settings := &fd.Settings{
// 		Formula: fd.Central, // Центральные разности для большей точности
// 		Step:    1e-6,       // Шаг дифференцирования

// 	}

// 	// Вычисляем градиент численно
// 	fd.Gradient(grad, Func, x, settings)
// 	//o.logger.Debug("Gradient", zap.Float64s("gradient", grad))
// 	//o.logger.Debug("X", zap.Float64s("X", x))
// }
// problem := optimize.Problem{
// 	Func: Func,
// 	Grad: Grad,
// }
