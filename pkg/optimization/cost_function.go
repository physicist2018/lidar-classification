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

func calcPenaltyForNegativeValues(x []float64) float64 {
	var penalty float64
	for _, val := range x {
		if val < 0 {
			penalty += 1000 * math.Pow(val, 2)
		}
		// if val > 1 {
		// 	penalty += 10 * math.Pow(val-1, 2)
		// }
	}
	//fmt.Println(penalty)
	return penalty
}

func calcSmoothPenalty(x []float64) float64 {
	var penalty float64
	eq := 0.0
	for _, val := range x {
		eq += val
	}
	eq -= 1.0
	if math.Abs(eq) > 0.01 {
		penalty += 1e4 * math.Pow(eq, 2)
	}
	return penalty
}

// calculateResidual вычисляет основную невязку без штрафов за ограничения.
// Возвращает: невязка (если критичные условия не выполнены).
func (c *CostFunction) calculateResidual(x []float64) float64 {
	nd, nu, ns, nw := x[0], x[1], x[2], x[3]

	// Вычисление объёмов
	vd := nd * c.conf.LR.D * c.conf.CV.D
	vu := nu * c.conf.LR.U * c.conf.CV.U
	vs := ns * c.conf.LR.S * c.conf.CV.S
	vw := nw * c.conf.LR.W * c.conf.CV.W
	vTotal := vd + vu + vs + vw

	// Уравнения (остатки)
	eq1 := nd + nu + ns + nw - 1.0
	eq2 := c.deltaPrimeD*nd + c.deltaPrimeU*nu +
		c.deltaPrimeS*ns + c.deltaPrimeW*nw - c.data.DeltaPrime
	eq3 := c.params.GfD*nd + c.params.GfU*nu +
		c.params.GfS*ns + c.params.GfW*nw - c.data.Gf

	var eq4 float64
	if vTotal > 1e-8 {
		mCalc := (c.params.MreD*vd + c.params.MreU*vu +
			c.params.MreS*vs + c.params.MreW*vw) / vTotal
		eq4 = mCalc - c.data.M
	} else {
		eq4 = 0
	}

	// Адаптивные веса
	weight1 := 1.0
	weight2 := 1.0 / math.Max(1e-8, math.Abs(c.data.DeltaPrime))
	weight3 := 1.0 / math.Max(1e-8, math.Abs(c.data.Gf))
	weight4 := 1.0 / math.Max(1e-8, math.Abs(c.data.M))

	// Нормированные остатки
	eps1 := weight1 * eq1
	eps2 := weight2 * eq2
	eps3 := weight3 * eq3
	eps4 := weight4 * eq4

	residual := math.Sqrt(eps1*eps1 + eps2*eps2 + eps3*eps3 + eps4*eps4)

	return residual
}

// Value — основная функция стоимости. Проверяет ограничения и добавляет штрафы.
func (c *CostFunction) Value(x []float64) float64 {
	// Проверка размерности
	if len(x) != 4 {
		return 1e10
	}

	nd, nu, ns, nw := x[0], x[1], x[2], x[3]

	c.logger.Info("Input",
		zap.Float64("nd", nd),
		zap.Float64("nu", nu),
		zap.Float64("ns", ns),
		zap.Float64("nw", nw),
		zap.Float64("mrD", c.params.MreD),
		zap.Float64("mrU", c.params.MreU),
		zap.Float64("mrS", c.params.MreS),
		zap.Float64("mrW", c.params.MreW))

	// Плавное наказание за отрицательность
	penalty := calcPenaltyForNegativeValues(x)

	// Вычисляем основную невязку
	residual := c.calculateResidual(x)

	// smoothPenalty
	smoothPenalty := calcSmoothPenalty(x)

	// Итоговый результат
	total := residual + penalty + smoothPenalty

	c.logger.Debug("Residuals",
		zap.Float64("residual", residual),
		zap.Float64("penalty", penalty),
		zap.Float64("smPenalty", smoothPenalty),
		zap.Float64("total", total))

	// if penalty > 0 {
	// 	return penalty
	// }

	return total
}

// Gradient вычисляет градиент функции в точке x и сохраняет результат в gradient
func (c *CostFunction) Gradient(x []float64) []float64 {
	n := len(x)
	gradient := make([]float64, n)
	h := 1e-3 // Лучше для численной устойчивости

	fx := c.Value(x)

	// Временный срез для модификаций (не трогаем входной x)
	xMod := make([]float64, n)
	copy(xMod, x)

	for i := 0; i < n; i++ {
		// Увеличиваем i-ю координату
		xMod[i] += h
		fxh := c.Value(xMod)

		// Вычисляем разностную производную
		if math.IsInf(fx, 0) || math.IsInf(fxh, 0) || math.IsNaN(fxh-fx) {
			gradient[i] = 0 // Или другое безопасное значение
		} else {
			gradient[i] = (fxh - fx) / h
		}

		// Восстанавливаем исходное значение
		xMod[i] = x[i]
	}

	c.logger.Debug("Gradient computed",
		zap.Float64s("input", x),
		zap.Float64("base_value", fx),
		zap.Float64s("gradient", gradient))

	return gradient
}
