import time
import json
from datetime import datetime
import pytz
from typing import Dict, List, Any
import statistics
from pathlib import Path

class ModelComparison:
    def __init__(self, version_manager=None, langsmith=None):
        self.version_manager = version_manager
        self.langsmith = langsmith
        self.results = []
        
    def compare_models(self, questions: List[str], model_configs: List[Dict], qa_chain_factory):
        """여러 모델로 같은 질문들에 대해 답변을 비교"""
        
        if self.version_manager:
            self.version_manager.logger.info(f"=== 모델 비교 시작 ({len(model_configs)}개 모델, {len(questions)}개 질문) ===")
        
        # 각 질문에 대해 모든 모델의 답변 수집
        for i, question in enumerate(questions):
            question_results = {
                "question_id": i + 1,
                "question": question,
                "timestamp": datetime.now(pytz.timezone('Asia/Seoul')).isoformat(),
                "models": []
            }
            
            # 각 모델로 답변 생성
            for model_config in model_configs:
                model_result = self._test_single_model(question, model_config, qa_chain_factory)
                question_results["models"].append(model_result)
                
                # 진행상황 출력
                print(f"질문 {i+1}/{len(questions)}: {model_config['name']} 완료")
            
            self.results.append(question_results)
            
        # 비교 결과 분석
        analysis = self._analyze_results()
        
        # 결과 저장
        self._save_results(analysis)
        
        return analysis
    
    def _test_single_model(self, question: str, model_config: Dict, qa_chain_factory) -> Dict:
        """단일 모델로 질문에 답변"""
        
        start_time = time.time()
        
        try:
            # QA 체인 생성
            qa_chain = qa_chain_factory(model_config)
            
            # 답변 생성
            response = qa_chain.invoke(question)
            
            # 실행 시간 계산
            execution_time = time.time() - start_time
            
            result = {
                "model_name": model_config['name'],
                "model_provider": model_config['provider'],
                "model_id": model_config['model_name'],
                "temperature": model_config.get('temperature', 0.7),
                "response": response,
                "execution_time": execution_time,
                "success": True,
                "error": None,
                "response_length": len(str(response)),
                "tokens_estimated": self._estimate_tokens(str(response))
            }
            
            # 로깅
            if self.version_manager:
                self.version_manager.logger.info(f"[{model_config['name']}] 실행시간: {execution_time:.2f}초")
                
        except Exception as e:
            execution_time = time.time() - start_time
            result = {
                "model_name": model_config['name'],
                "model_provider": model_config['provider'],
                "model_id": model_config['model_name'],
                "temperature": model_config.get('temperature', 0.7),
                "response": None,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "response_length": 0,
                "tokens_estimated": 0
            }
            
            if self.version_manager:
                self.version_manager.logger.error(f"[{model_config['name']}] 오류: {e}")
        
        return result
    
    def _estimate_tokens(self, text: str) -> int:
        """토큰 수 추정 (대략적)"""
        # 간단한 추정: 단어 수의 1.3배 정도
        words = len(text.split())
        return int(words * 1.3)
    
    def _analyze_results(self) -> Dict:
        """결과 분석"""
        
        analysis = {
            "summary": {
                "total_questions": len(self.results),
                "total_models": len(self.results[0]["models"]) if self.results else 0,
                "timestamp": datetime.now(pytz.timezone('Asia/Seoul')).isoformat()
            },
            "model_performance": {},
            "detailed_comparison": self.results
        }
        
        if not self.results:
            return analysis
        
        # 모델별 성능 통계
        model_names = [model["model_name"] for model in self.results[0]["models"]]
        
        for model_name in model_names:
            model_data = []
            success_count = 0
            total_time = 0
            total_length = 0
            total_tokens = 0
            
            # 각 질문의 해당 모델 결과 수집
            for question_result in self.results:
                for model_result in question_result["models"]:
                    if model_result["model_name"] == model_name:
                        model_data.append(model_result)
                        if model_result["success"]:
                            success_count += 1
                            total_time += model_result["execution_time"]
                            total_length += model_result["response_length"]
                            total_tokens += model_result["tokens_estimated"]
            
            # 성능 메트릭 계산
            analysis["model_performance"][model_name] = {
                "success_rate": success_count / len(model_data) if model_data else 0,
                "average_response_time": total_time / success_count if success_count > 0 else 0,
                "average_response_length": total_length / success_count if success_count > 0 else 0,
                "average_tokens": total_tokens / success_count if success_count > 0 else 0,
                "total_questions": len(model_data),
                "successful_responses": success_count,
                "failed_responses": len(model_data) - success_count
            }
        
        return analysis
    
    def _save_results(self, analysis: Dict):
        """결과를 파일로 저장"""
        
        # 결과 디렉토리 생성
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # 한국시각 기준 파일명 생성
        kst = pytz.timezone('Asia/Seoul')
        now = datetime.now(kst)
        timestamp = now.strftime("%m%d%H%M")
        
        # JSON 파일로 저장
        json_file = results_dir / f"model_comparison_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        # 사람이 읽기 쉬운 리포트 생성
        report_file = results_dir / f"model_comparison_report_{timestamp}.md"
        self._generate_report(analysis, report_file)
        
        if self.version_manager:
            self.version_manager.logger.info(f"비교 결과 저장: {json_file}")
            self.version_manager.logger.info(f"리포트 생성: {report_file}")
    
    def _generate_report(self, analysis: Dict, report_file: Path):
        """사람이 읽기 쉬운 마크다운 리포트 생성"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 🤖 LLM 모델 비교 리포트\n\n")
            
            # 요약 정보
            summary = analysis["summary"]
            f.write(f"**생성 시간**: {summary['timestamp']}\n")
            f.write(f"**총 질문 수**: {summary['total_questions']}\n")
            f.write(f"**비교 모델 수**: {summary['total_models']}\n\n")
            
            # 성능 비교표
            f.write("## 📊 모델별 성능 비교\n\n")
            f.write("| 모델 | 성공률 | 평균 응답시간 | 평균 응답길이 | 평균 토큰수 |\n")
            f.write("|------|---------|---------------|---------------|-------------|\n")
            
            for model_name, perf in analysis["model_performance"].items():
                f.write(f"| {model_name} | {perf['success_rate']:.1%} | {perf['average_response_time']:.2f}s | {perf['average_response_length']:.0f} | {perf['average_tokens']:.0f} |\n")
            
            f.write("\n## 📝 상세 질문별 비교\n\n")
            
            # 각 질문별 상세 결과
            for i, question_result in enumerate(analysis["detailed_comparison"]):
                f.write(f"### 질문 {i+1}\n")
                f.write(f"**질문**: {question_result['question']}\n\n")
                
                for model_result in question_result["models"]:
                    f.write(f"#### {model_result['model_name']}\n")
                    if model_result["success"]:
                        f.write(f"**응답**: {model_result['response']}\n")
                        f.write(f"**실행시간**: {model_result['execution_time']:.2f}초\n")
                        f.write(f"**응답길이**: {model_result['response_length']}자\n\n")
                    else:
                        f.write(f"**오류**: {model_result['error']}\n\n")
                
                f.write("---\n\n")
        
        print(f"\n📊 상세 비교 리포트가 생성되었습니다: {report_file}")