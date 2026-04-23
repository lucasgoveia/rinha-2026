package handler

import (
	"net/http"

	"github.com/labstack/echo/v4"
	"github.com/lucasgoveia/rinha2026/internal/fraud"
)

type Handler struct {
	Refs    *fraud.References
	MccRisk map[string]float64
	Norm    fraud.NormConstants
}

type fraudScoreResponse struct {
	Approved   bool    `json:"approved"`
	FraudScore float64 `json:"fraud_score"`
}

func (h *Handler) Ready(c echo.Context) error {
	return c.NoContent(http.StatusOK)
}

func (h *Handler) FraudScore(c echo.Context) error {
	var req fraud.FraudScoreRequest
	if err := c.Bind(&req); err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{"error": err.Error()})
	}

	vec := fraud.Vectorize(&req, h.MccRisk, h.Norm)
	score, approved := fraud.Score(vec, h.Refs)

	return c.JSON(http.StatusOK, fraudScoreResponse{
		Approved:   approved,
		FraudScore: score,
	})
}
