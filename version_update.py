#!/usr/bin/env python3
"""
버전 관리 및 시스템 업데이트 스크립트
사용법: python version_update.py [description]
"""

import sys
import argparse
from pathlib import Path
from src.utils.version_manager import VersionManager

def main():
    parser = argparse.ArgumentParser(description='버전 관리 및 파일 백업')
    parser.add_argument('description', nargs='?', default='시스템 업데이트', 
                       help='변경 사항 설명')
    parser.add_argument('--file', '-f', help='특정 파일 버전 생성')
    parser.add_argument('--backup-all', action='store_true', 
                       help='전체 프로젝트 백업')
    
    args = parser.parse_args()
    
    vm = VersionManager()
    
    if args.backup_all:
        # 전체 프로젝트 백업
        project_root = Path(__file__).parent
        backup_path = vm.backup_directory(project_root, args.description)
        print(f"전체 프로젝트 백업 완료: {backup_path}")
    
    elif args.file:
        # 특정 파일 버전 생성
        try:
            versioned_path = vm.create_versioned_file(args.file, args.description)
            print(f"파일 버전 생성 완료: {versioned_path}")
        except FileNotFoundError as e:
            print(f"오류: {e}")
            return 1
    
    else:
        # 주요 파일들 버전 생성
        important_files = [
            'src/main.py',
            'src/utils/document_loaders.py',
            'src/components/vectorstores.py',
            'conf/config.yaml'
        ]
        
        for file_path in important_files:
            try:
                versioned_path = vm.create_versioned_file(file_path, args.description)
                print(f"버전 생성: {versioned_path}")
            except FileNotFoundError:
                print(f"파일을 찾을 수 없음: {file_path}")
    
    print("버전 관리 작업이 완료되었습니다.")
    print(f"로그 확인: logs/version_history.log")
    return 0

if __name__ == "__main__":
    sys.exit(main())