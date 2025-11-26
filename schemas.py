from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator

class Party(BaseModel):
    name: str
    role: Literal["被告", "原告", "上诉人", "被上诉人", "第三人", "公诉机关"]
    details: Optional[str] = Field(None, description="如果有的话，包括年龄、民族或住址")

class Charge(BaseModel):
    crime_name: str = Field(..., description="罪名，例如 '盗窃罪'、'故意伤害罪'")
    facts: str = Field(..., description="支持该指控的事实简述")
    verdict: Literal["有罪", "无罪", "驳回"]
    sentence: Optional[str] = Field(None, description="该指控的具体判决，例如 '有期徒刑六个月'")

class JudgmentExtraction(BaseModel):
    court_name: str = Field(..., description="发布判决的法院名称")
    case_number: str = Field(..., description="唯一的案号，例如 '(2024)京0105刑初1234号'")
    date: str = Field(..., description="判决日期")
    
    parties: List[Party]
    charges: List[Charge]
    
    # 思维链：强制模型在给出最终总结前进行推理
    reasoning_process: str = Field(..., description="法官如何得出判决的逐步推理过程")
    
    final_outcome: str = Field(..., description="最终裁决的简明总结（总刑期、罚金等）")

    @field_validator('case_number')
    def validate_case_number(cls, v):
        if "号" not in v:
            raise ValueError("案号必须包含 '号'（标准中文格式）")
        return v
