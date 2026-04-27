package fraud

import "time"

type NormConstants struct {
	MaxAmount            float64 `json:"max_amount"`
	MaxInstallments      float64 `json:"max_installments"`
	AmountVsAvgRatio     float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float64 `json:"max_minutes"`
	MaxKm                float64 `json:"max_km"`
	MaxTxCount24h        float64 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float64 `json:"max_merchant_avg_amount"`
}

type TransactionReq struct {
	Amount      float64 `json:"amount"`
	Installments int    `json:"installments"`
	RequestedAt string  `json:"requested_at"`
}

type CustomerReq struct {
	AvgAmount       float64  `json:"avg_amount"`
	TxCount24h      int      `json:"tx_count_24h"`
	KnownMerchants  []string `json:"known_merchants"`
}

type MerchantReq struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float64 `json:"avg_amount"`
}

type TerminalReq struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float64 `json:"km_from_home"`
}

type LastTransactionReq struct {
	Timestamp      string  `json:"timestamp"`
	KmFromCurrent  float64 `json:"km_from_current"`
}

type FraudScoreRequest struct {
	ID              string              `json:"id"`
	Transaction     TransactionReq      `json:"transaction"`
	Customer        CustomerReq         `json:"customer"`
	Merchant        MerchantReq         `json:"merchant"`
	Terminal        TerminalReq         `json:"terminal"`
	LastTransaction *LastTransactionReq `json:"last_transaction"`
}

func clamp(x float64) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return float32(x)
}

func boolF32(b bool) float32 {
	if b {
		return 1
	}
	return 0
}

func Vectorize(req *FraudScoreRequest, mccRisk map[string]float64, norm NormConstants) [16]float32 {
	t, _ := time.Parse(time.RFC3339, req.Transaction.RequestedAt)
	hour := float64(t.UTC().Hour())
	wdRaw := int(t.UTC().Weekday()+6) % 7
	dow := float64(wdRaw) / 6

	mccRiskVal, ok := mccRisk[req.Merchant.MCC]
	if !ok {
		mccRiskVal = 0.5
	}

	unknownMerchant := 1.0
	for _, m := range req.Customer.KnownMerchants {
		if m == req.Merchant.ID {
			unknownMerchant = 0
			break
		}
	}

	var minutesSinceLast, kmFromLast float32
	if req.LastTransaction == nil {
		minutesSinceLast = -1
		kmFromLast = -1
	} else {
		lastT, _ := time.Parse(time.RFC3339, req.LastTransaction.Timestamp)
		diff := t.Sub(lastT).Minutes()
		minutesSinceLast = clamp(diff / norm.MaxMinutes)
		kmFromLast = clamp(req.LastTransaction.KmFromCurrent / norm.MaxKm)
	}

	return [16]float32{
		clamp(req.Transaction.Amount / norm.MaxAmount),
		clamp(float64(req.Transaction.Installments) / norm.MaxInstallments),
		clamp((req.Transaction.Amount / req.Customer.AvgAmount) / norm.AmountVsAvgRatio),
		float32(hour / 23),
		float32(dow),
		minutesSinceLast,
		kmFromLast,
		clamp(req.Terminal.KmFromHome / norm.MaxKm),
		clamp(float64(req.Customer.TxCount24h) / norm.MaxTxCount24h),
		boolF32(req.Terminal.IsOnline),
		boolF32(req.Terminal.CardPresent),
		float32(unknownMerchant),
		float32(mccRiskVal),
		clamp(req.Merchant.AvgAmount / norm.MaxMerchantAvgAmount),
		0, // padding
		0, // padding
	}
}
