package domain

// FileReader интерфейс для чтения файлов
type FileReader interface {
	ReadMatrix(filename string) (*MatrixData, error)
}

// FileWriter интерфейс для записи результатов
type FileWriter interface {
	WriteMatrix(filename string, data *MatrixData) error
}

// ConfigReader интерфейс для чтения конфигурации
type ConfigReader interface {
	ReadConfig(path string) (*Config, error)
}

type Historgammer interface {
	Hist(min, max float64, n int) ([]float64, []float64, error)
}
