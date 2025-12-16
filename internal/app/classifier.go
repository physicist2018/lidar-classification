package app

import (
	"lidar-classification/internal/domain"
	"lidar-classification/pkg/optimization"
	"math"
	"sync"

	"go.uber.org/zap"
)

type AerosolClassifier struct {
	logger    *zap.Logger
	optimizer *optimization.MonteCarloOptimizer
	config    *domain.Config
}

func NewAerosolClassifier(logger *zap.Logger, config *domain.Config) *AerosolClassifier {
	return &AerosolClassifier{
		logger:    logger,
		optimizer: optimization.NewMonteCarloOptimizer(logger),
		config:    config,
	}
}

func (c *AerosolClassifier) ProcessMatrices(depData, flData, mreData *domain.MatrixData) map[string]*domain.MatrixData {
	results := c.initializeResultMatrices(depData.Rows, depData.Cols)

	var wg sync.WaitGroup
	taskChan := make(chan domain.ProcessingTask, c.config.Workers*2)
	resultChan := make(chan *domain.ProcessingResult, depData.Rows*depData.Cols)

	// Запускаем воркеры
	for i := range c.config.Workers {
		wg.Add(1)
		c.logger.Info("Starting worker", zap.Int("id", i))
		go c.worker(i, taskChan, resultChan, &wg)
	}

	// Отправляем задачи
	go func() {
		for i := range depData.Rows {
			for j := range depData.Cols {
				pointData := c.preparePointData(i, j, depData, flData, mreData)
				if c.validatePointData(pointData) {
					task := domain.ProcessingTask{
						I:      i,
						J:      j,
						Data:   pointData,
						Config: c.config,
						Result: resultChan,
					}
					taskChan <- task
				}
			}
		}
		close(taskChan)
	}()

	// Собираем результаты
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Обрабатываем результаты
	for result := range resultChan {
		if result.Solution.IsValid {
			c.updateResults(results, result)
		}
	}

	return results
}

func (c *AerosolClassifier) worker(id int, tasks <-chan domain.ProcessingTask, results chan<- *domain.ProcessingResult, wg *sync.WaitGroup) {
	defer wg.Done()

	for task := range tasks {
		c.logger.Debug("Processing point",
			zap.Int("worker", id),
			zap.Int("i", task.I),
			zap.Int("j", task.J))

		solution := c.optimizer.Solve(task.Data, task.Config)

		results <- &domain.ProcessingResult{
			I:        task.I,
			J:        task.J,
			Solution: solution,
		}
	}
}

func (c *AerosolClassifier) preparePointData(i, j int, dep, fl, mre *domain.MatrixData) *domain.PointData {
	delta := dep.Data[i][j] / 100.0 // Конвертируем проценты
	deltaPrime := delta / (1 + delta)

	return &domain.PointData{
		I:          i,
		J:          j,
		DeltaPrime: deltaPrime,
		Gf:         fl.Data[i][j],
		M:          mre.Data[i][j],
	}
}

func (c *AerosolClassifier) validatePointData(data *domain.PointData) bool {
	return (!math.IsNaN(data.DeltaPrime) && !math.IsNaN(data.Gf) && !math.IsNaN(data.M)) &&
		data.DeltaPrime > 0 && data.Gf > 0 && data.M > 0
}

func (c *AerosolClassifier) initializeResultMatrices(rows, cols int) map[string]*domain.MatrixData {
	matrices := make(map[string]*domain.MatrixData)
	outputFiles := []string{
		"residuals", "n_d", "n_u", "n_s", "n_w",
		"GF_d", "GF_u", "GF_s", "GF_w",
		"delta_d", "delta_u", "delta_s", "delta_w",
		"mre_d", "mre_u", "mre_s", "mre_w",
		"diff_eq1", "diff_eq2", "diff_eq3", "diff_eq4",
	}

	for _, name := range outputFiles {
		data := make([][]float64, rows)
		for i := range data {
			data[i] = make([]float64, cols)
			for j := range data[i] {
				data[i][j] = math.NaN() // Значение для необработанных точек
			}
		}
		matrices[name] = &domain.MatrixData{
			Data: data,
			Rows: rows,
			Cols: cols,
		}
	}
	return matrices
}

func (c *AerosolClassifier) updateResults(results map[string]*domain.MatrixData, result *domain.ProcessingResult) {
	i, j := result.I, result.J
	sol := result.Solution

	results["residuals"].Data[i][j] = sol.Residual
	results["n_d"].Data[i][j] = sol.Fractions.D
	results["n_u"].Data[i][j] = sol.Fractions.U
	results["n_s"].Data[i][j] = sol.Fractions.S
	results["n_w"].Data[i][j] = sol.Fractions.W
	results["GF_d"].Data[i][j] = sol.Parameters.GfD
	results["GF_u"].Data[i][j] = sol.Parameters.GfU
	results["GF_s"].Data[i][j] = sol.Parameters.GfS
	results["GF_w"].Data[i][j] = sol.Parameters.GfW
	results["delta_d"].Data[i][j] = sol.Parameters.DeltaD / (1 - sol.Parameters.DeltaD)
	results["delta_u"].Data[i][j] = sol.Parameters.DeltaU / (1 - sol.Parameters.DeltaU)
	results["delta_s"].Data[i][j] = sol.Parameters.DeltaS / (1 - sol.Parameters.DeltaS)
	results["delta_w"].Data[i][j] = sol.Parameters.DeltaW / (1 - sol.Parameters.DeltaW)
	results["mre_d"].Data[i][j] = sol.Parameters.MreD
	results["mre_u"].Data[i][j] = sol.Parameters.MreU
	results["mre_s"].Data[i][j] = sol.Parameters.MreS
	results["mre_w"].Data[i][j] = sol.Parameters.MreW
	results["diff_eq1"].Data[i][j] = sol.Difference[0]
	results["diff_eq2"].Data[i][j] = sol.Difference[1]
	results["diff_eq3"].Data[i][j] = sol.Difference[2]
	results["diff_eq4"].Data[i][j] = sol.Difference[3]

}
