package domain

// ClassificationService сервис классификации аэрозолей
type ClassificationService interface {
	ProcessPoint(data *PointData, config *Config) *Solution
	ValidatePoint(data *PointData) bool
}

// WorkerPool интерфейс пула воркеров
type WorkerPool interface {
	Start()
	Submit(task *ProcessingTask)
	Stop()
}

// ProcessingTask задача обработки точки
type ProcessingTask struct {
	I, J   int
	Data   *PointData
	Config *Config
	Result chan<- *ProcessingResult
}

type ProcessingResult struct {
	I, J     int
	Solution *Solution
}
