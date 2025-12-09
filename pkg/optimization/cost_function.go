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

func (c *CostFunction) Value(x []float64) float64 {
	// x = [n_d, n_u, n_s, n_w]
	if len(x) != 4 {
		return math.Inf(1)
	}

	nd, nu, ns, nw := x[0], x[1], x[2], x[3]

	// Ограничения неотрицательности
	if nd < 0 || nu < 0 || ns < 0 || nw < 0 {
		return HUGE_VAL
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
	if vTotal > 0 {
		mCalc := (c.conf.M.D*vd + c.conf.M.U*vu + c.conf.M.S*vs + c.conf.M.W*vw) / vTotal
		eq4 = mCalc - c.data.M
	} else {
		eq4 = HUGE_VAL
	}

	// Невязка как норма Фробениуса относительных отклонений
	residual := math.Sqrt(
		math.Pow(eq1/1.0, 2) +
			math.Pow(eq2/c.data.DeltaPrime, 2) +
			math.Pow(eq3/c.data.Gf, 2) +
			math.Pow(eq4/c.data.M, 2))
	c.logger.Debug("Residual", zap.Float64("", residual*100))
	return residual
}

// // Grad вычисляет градиент функции в точке x и сохраняет результат в grad
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
