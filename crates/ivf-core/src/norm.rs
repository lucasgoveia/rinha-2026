use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct NormalizationConfig {
    pub max_amount: f32,
    pub max_installments: f32,
    pub amount_vs_avg_ratio: f32,
    pub max_minutes: f32,
    pub max_km: f32,
    pub max_tx_count_24h: f32,
    pub max_merchant_avg_amount: f32,
}

impl NormalizationConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct MerchantRiskConfig(HashMap<String, f32>);

impl MerchantRiskConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn risk(&self, mcc: &str) -> f32 {
        *self.0.get(mcc).unwrap_or(&0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_normalization_config() {
        let config = NormalizationConfig::load("../../resources/normalization.json").unwrap();
        assert_eq!(config.max_amount, 10000.0);
        assert_eq!(config.max_installments, 12.0);
        assert_eq!(config.amount_vs_avg_ratio, 10.0);
        assert_eq!(config.max_minutes, 1440.0);
        assert_eq!(config.max_km, 1000.0);
        assert_eq!(config.max_tx_count_24h, 20.0);
        assert_eq!(config.max_merchant_avg_amount, 10000.0);
    }

    #[test]
    fn load_merchant_risk_config() {
        let config = MerchantRiskConfig::load("../../resources/mcc_risk.json").unwrap();
        assert_eq!(config.risk("5411"), 0.15);
        assert_eq!(config.risk("7995"), 0.85);
        assert_eq!(config.risk("9999"), 0.5); // unknown → default 0.5
    }
}
