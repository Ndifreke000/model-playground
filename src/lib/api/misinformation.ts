/**
 * Misinformation Detection API Client
 * TypeScript client for the Python ML backend
 */

const API_BASE_URL = import.meta.env.VITE_ML_API_URL || 'http://localhost:8000';

export interface PredictionResult {
    prediction: 'fake' | 'real';
    confidence: number;
    raw_score: number;
}

export interface BatchPredictionResult {
    results: PredictionResult[];
}

export interface HealthStatus {
    status: 'healthy' | 'unhealthy';
    model_loaded: boolean;
    device: string;
}

export class MisinformationAPIError extends Error {
    constructor(
        message: string,
        public status?: number,
        public details?: unknown
    ) {
        super(message);
        this.name = 'MisinformationAPIError';
    }
}

class MisinformationAPI {
    private baseURL: string;

    constructor(baseURL: string = API_BASE_URL) {
        this.baseURL = baseURL;
    }

    /**
     * Check API health status
     */
    async healthCheck(): Promise<HealthStatus> {
        try {
            const response = await fetch(`${this.baseURL}/health`);

            if (!response.ok) {
                throw new MisinformationAPIError(
                    'Health check failed',
                    response.status
                );
            }

            return await response.json();
        } catch (error) {
            if (error instanceof MisinformationAPIError) {
                throw error;
            }
            throw new MisinformationAPIError(
                'Failed to connect to ML service. Is the server running?',
                undefined,
                error
            );
        }
    }

    /**
     * Predict if a single text is fake news
     */
    async predict(text: string): Promise<PredictionResult> {
        if (!text || text.trim().length < 10) {
            throw new MisinformationAPIError('Text is too short (minimum 10 characters)');
        }

        if (text.length > 10000) {
            throw new MisinformationAPIError('Text is too long (maximum 10000 characters)');
        }

        try {
            const response = await fetch(`${this.baseURL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new MisinformationAPIError(
                    errorData.detail || 'Prediction failed',
                    response.status,
                    errorData
                );
            }

            return await response.json();
        } catch (error) {
            if (error instanceof MisinformationAPIError) {
                throw error;
            }
            throw new MisinformationAPIError(
                'Failed to get prediction',
                undefined,
                error
            );
        }
    }

    /**
     * Predict multiple texts at once
     */
    async batchPredict(texts: string[]): Promise<BatchPredictionResult> {
        if (!texts || texts.length === 0) {
            throw new MisinformationAPIError('No texts provided');
        }

        if (texts.length > 100) {
            throw new MisinformationAPIError('Too many texts (maximum 100 per batch)');
        }

        try {
            const response = await fetch(`${this.baseURL}/batch_predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ texts }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new MisinformationAPIError(
                    errorData.detail || 'Batch prediction failed',
                    response.status,
                    errorData
                );
            }

            return await response.json();
        } catch (error) {
            if (error instanceof MisinformationAPIError) {
                throw error;
            }
            throw new MisinformationAPIError(
                'Failed to get batch predictions',
                undefined,
                error
            );
        }
    }
}

// Export singleton instance
export const misinformationAPI = new MisinformationAPI();
