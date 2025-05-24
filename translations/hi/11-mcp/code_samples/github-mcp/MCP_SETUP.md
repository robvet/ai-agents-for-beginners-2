<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "c4be907703b836d1a1c360db20da4de9",
  "translation_date": "2025-05-21T08:16:33+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/MCP_SETUP.md",
  "language_code": "hi"
}
-->
# MCP Server इंटीग्रेशन गाइड

## आवश्यकताएँ
- Node.js इंस्टॉल हो (संस्करण 14 या उससे ऊपर)
- npm पैकेज मैनेजर
- Python एनवायरनमेंट जिसमें जरूरी डिपेंडेंसीज़ हों

## सेटअप स्टेप्स

1. **MCP Server पैकेज इंस्टॉल करें**
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

2. **MCP Server शुरू करें**
   ```bash
   npx @modelcontextprotocol/server-github
   ```  
   सर्वर शुरू हो जाएगा और कनेक्शन URL दिखाएगा।

3. **कनेक्शन सत्यापित करें**
   - अपने Chainlit इंटरफ़ेस में प्लग आइकन (🔌) देखें  
   - प्लग आइकन के पास संख्या (1) दिखनी चाहिए, जो सफल कनेक्शन दर्शाती है  
   - कंसोल में दिखना चाहिए: "GitHub plugin setup completed successfully" (साथ में अन्य स्टेटस लाइनें भी हो सकती हैं)

## समस्या निवारण

### सामान्य समस्याएं

1. **पोर्ट संघर्ष**
   ```bash
   Error: listen EADDRINUSE: address already in use
   ```  
   समाधान: पोर्ट बदलने के लिए यह कमांड इस्तेमाल करें:  
   ```bash
   npx @modelcontextprotocol/server-github --port 3001
   ```

2. **प्रमाणीकरण समस्याएं**
   - सुनिश्चित करें कि GitHub क्रेडेंशियल सही तरीके से कॉन्फ़िगर किए गए हैं  
   - .env फाइल में आवश्यक टोकन मौजूद हों  
   - GitHub API एक्सेस की जांच करें

3. **कनेक्शन फेल**
   - पुष्टि करें कि सर्वर अपेक्षित पोर्ट पर चल रहा है  
   - फ़ायरवॉल सेटिंग्स चेक करें  
   - Python एनवायरनमेंट में जरूरी पैकेज इंस्टॉल हों

## कनेक्शन सत्यापन

आपका MCP सर्वर सही तरीके से जुड़ा होगा जब:  
1. कंसोल में "GitHub plugin setup completed successfully" दिखे  
2. कनेक्शन लॉग में "✓ MCP Connection Status: Active" दिखाई दे  
3. चैट इंटरफ़ेस में GitHub कमांड काम करें

## एनवायरनमेंट वेरिएबल्स

आपकी .env फाइल में ये जरूरी हैं:  
```
GITHUB_TOKEN=your_github_token
MCP_SERVER_PORT=3000  # Optional, default is 3000
```

## कनेक्शन टेस्टिंग

चैट में यह टेस्ट मेसेज भेजें:  
```
Show me the repositories for username: [GitHub Username]
```  
सफल प्रतिक्रिया में रिपॉजिटरी की जानकारी दिखेगी।

**अस्वीकरण**:  
यह दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके अनुवादित किया गया है। जबकि हम सटीकता के लिए प्रयासरत हैं, कृपया ध्यान दें कि स्वचालित अनुवादों में त्रुटियाँ या असंगतियाँ हो सकती हैं। मूल दस्तावेज़ को इसकी मूल भाषा में प्रामाणिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सलाह दी जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम जिम्मेदार नहीं हैं।