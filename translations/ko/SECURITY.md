<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "d8fe220fa2850df0759b07cf391ea77c",
  "translation_date": "2025-05-20T08:32:06+00:00",
  "source_file": "SECURITY.md",
  "language_code": "ko"
}
-->
# 보안

마이크로소프트는 당사 소프트웨어 제품과 서비스의 보안을 매우 중요하게 여기며, 여기에는 [Microsoft](https://github.com/Microsoft), [Azure](https://github.com/Azure), [DotNet](https://github.com/dotnet), [AspNet](https://github.com/aspnet), [Xamarin](https://github.com/xamarin) 등 당사 GitHub 조직에서 관리하는 모든 소스 코드 저장소가 포함됩니다.

만약 마이크로소프트 소유 저장소에서 [마이크로소프트의 보안 취약점 정의](https://aka.ms/security.md/definition)에 부합하는 보안 취약점을 발견하셨다면, 아래 설명에 따라 신고해 주시기 바랍니다.

## 보안 문제 신고

**보안 취약점은 공개 GitHub 이슈를 통해 신고하지 말아 주세요.**

대신, Microsoft Security Response Center(MSRC)에서 [https://msrc.microsoft.com/create-report](https://aka.ms/security.md/msrc/create-report)로 신고해 주시기 바랍니다.

로그인 없이 제출하시려면 [secure@microsoft.com](mailto:secure@microsoft.com)으로 이메일을 보내 주세요. 가능하다면, 당사 PGP 키로 메시지를 암호화해 주시기 바랍니다. 키는 [Microsoft Security Response Center PGP Key 페이지](https://aka.ms/security.md/msrc/pgp)에서 다운로드할 수 있습니다.

보통 24시간 이내에 답변을 받으실 수 있습니다. 만약 답변을 받지 못하셨다면, 이메일로 다시 연락해 주셔서 원본 메시지를 잘 받았는지 확인해 주세요. 추가 정보는 [microsoft.com/msrc](https://www.microsoft.com/msrc)에서 확인할 수 있습니다.

아래 요청된 정보를 최대한 포함해 주시면 문제의 성격과 범위를 더 잘 이해하는 데 도움이 됩니다:

* 문제 유형(예: 버퍼 오버플로우, SQL 인젝션, 크로스 사이트 스크립팅 등)
* 문제 발생과 관련된 소스 파일의 전체 경로
* 영향을 받는 소스 코드 위치(tag/branch/commit 또는 직접 URL)
* 문제 재현에 필요한 특별한 설정
* 문제 재현 단계별 지침
* 개념 증명 또는 익스플로잇 코드(가능한 경우)
* 문제의 영향 및 공격자가 이를 어떻게 악용할 수 있는지

이 정보는 신고 내용을 신속히 분류하는 데 큰 도움이 됩니다.

버그 바운티 신고인 경우, 보다 완성도 높은 보고서는 더 높은 보상으로 이어질 수 있습니다. 자세한 내용은 [Microsoft Bug Bounty Program](https://aka.ms/security.md/msrc/bounty) 페이지를 참조해 주세요.

## 선호 언어

모든 소통은 영어로 진행되기를 선호합니다.

## 정책

마이크로소프트는 [Coordinated Vulnerability Disclosure](https://aka.ms/security.md/cvd) 원칙을 따릅니다.

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 노력하고 있으나, 자동 번역에는 오류나 부정확한 부분이 있을 수 있음을 양지해 주시기 바랍니다. 원본 문서는 해당 언어의 원문이 권위 있는 자료로 간주되어야 합니다. 중요한 정보의 경우 전문적인 인간 번역을 권장합니다. 본 번역의 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.