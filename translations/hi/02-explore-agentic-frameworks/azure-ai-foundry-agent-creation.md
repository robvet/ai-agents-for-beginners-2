<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7e92870dc0843e13d4dabc620c09d2d9",
  "translation_date": "2025-05-20T09:41:38+00:00",
  "source_file": "02-explore-agentic-frameworks/azure-ai-foundry-agent-creation.md",
  "language_code": "hi"
}
-->
# Azure AI Agent Service Development

इस अभ्यास में, आप [Azure AI Foundry portal](https://ai.azure.com/?WT.mc_id=academic-105485-koreyst) में Azure AI Agent सेवा उपकरणों का उपयोग करके फ्लाइट बुकिंग के लिए एक एजेंट बनाएंगे। यह एजेंट उपयोगकर्ताओं के साथ बातचीत कर सकेगा और फ्लाइट्स के बारे में जानकारी प्रदान करेगा।

## आवश्यकताएँ

इस अभ्यास को पूरा करने के लिए, आपके पास निम्नलिखित होना चाहिए:
1. एक सक्रिय सदस्यता के साथ Azure खाता। [मुफ्त में खाता बनाएँ](https://azure.microsoft.com/free/?WT.mc_id=academic-105485-koreyst)।
2. Azure AI Foundry हब बनाने की अनुमति या आपके लिए पहले से बना हुआ हब।
    - यदि आपकी भूमिका Contributor या Owner है, तो आप इस ट्यूटोरियल के चरणों का पालन कर सकते हैं।

## Azure AI Foundry हब बनाएँ

> **Note:** Azure AI Foundry पहले Azure AI Studio के नाम से जाना जाता था।

1. Azure AI Foundry हब बनाने के लिए [Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-studio/?WT.mc_id=academic-105485-koreyst) ब्लॉग पोस्ट में दिए गए दिशानिर्देशों का पालन करें।
2. जब आपका प्रोजेक्ट बन जाए, तो दिखाई देने वाले किसी भी टिप्स को बंद करें और Azure AI Foundry पोर्टल में प्रोजेक्ट पेज की समीक्षा करें, जो इस छवि के समान दिखना चाहिए:

    ![Azure AI Foundry Project](../../../translated_images/azure-ai-foundry.8a2b56713298fd09de77022ab1ba07ebc681ea4cd4438a46c4a6fc6b6f077962.hi.png)

## मॉडल तैनात करें

1. अपने प्रोजेक्ट के बाएं पैन में, **My assets** सेक्शन में, **Models + endpoints** पेज चुनें।
2. **Models + endpoints** पेज में, **Model deployments** टैब पर, **+ Deploy model** मेनू में, **Deploy base model** चुनें।
3. सूची में `gpt-4o-mini` मॉडल खोजें, फिर उसे चुनें और पुष्टि करें।

    > **Note**: TPM कम करने से आपके उपयोग में सदस्यता के तहत उपलब्ध कोटा की अधिक खपत से बचा जा सकता है।

    ![Model Deployed](../../../translated_images/model-deployment.4adf429ebdf42103d7a759087fe0da91aeb70d2204cc8bdca70cc6c53c627938.hi.png)

## एजेंट बनाएँ

अब जब आपने मॉडल तैनात कर दिया है, तो आप एक एजेंट बना सकते हैं। एजेंट एक संवादात्मक AI मॉडल है जिसका उपयोग उपयोगकर्ताओं के साथ बातचीत के लिए किया जाता है।

1. अपने प्रोजेक्ट के बाएं पैन में, **Build & Customize** सेक्शन में, **Agents** पेज चुनें।
2. नया एजेंट बनाने के लिए **+ Create agent** पर क्लिक करें। **Agent Setup** संवाद बॉक्स में:
    - एजेंट के लिए एक नाम दर्ज करें, जैसे `FlightAgent`।
    - सुनिश्चित करें कि आपने पहले से बनाया हुआ `gpt-4o-mini` मॉडल तैनाती चुना हुआ है।
    - एजेंट को दिए जाने वाले निर्देश (Instructions) सेट करें। उदाहरण के लिए:
    ```
    You are FlightAgent, a virtual assistant specialized in handling flight-related queries. Your role includes assisting users with searching for flights, retrieving flight details, checking seat availability, and providing real-time flight status. Follow the instructions below to ensure clarity and effectiveness in your responses:

    ### Task Instructions:
    1. **Recognizing Intent**:
       - Identify the user's intent based on their request, focusing on one of the following categories:
         - Searching for flights
         - Retrieving flight details using a flight ID
         - Checking seat availability for a specified flight
         - Providing real-time flight status using a flight number
       - If the intent is unclear, politely ask users to clarify or provide more details.
        
    2. **Processing Requests**:
        - Depending on the identified intent, perform the required task:
        - For flight searches: Request details such as origin, destination, departure date, and optionally return date.
        - For flight details: Request a valid flight ID.
        - For seat availability: Request the flight ID and date and validate inputs.
        - For flight status: Request a valid flight number.
        - Perform validations on provided data (e.g., formats of dates, flight numbers, or IDs). If the information is incomplete or invalid, return a friendly request for clarification.

    3. **Generating Responses**:
    - Use a tone that is friendly, concise, and supportive.
    - Provide clear and actionable suggestions based on the output of each task.
    - If no data is found or an error occurs, explain it to the user gently and offer alternative actions (e.g., refine search, try another query).
    
    ```
> [!NOTE]
> विस्तृत प्रॉम्प्ट के लिए, आप [इस रिपॉजिटरी](https://github.com/ShivamGoyal03/RoamMind) को देख सकते हैं।

> इसके अलावा, आप एजेंट की क्षमताओं को बढ़ाने के लिए **Knowledge Base** और **Actions** जोड़ सकते हैं, ताकि उपयोगकर्ता की मांगों के आधार पर अधिक जानकारी प्रदान की जा सके और स्वचालित कार्य किए जा सकें। इस अभ्यास के लिए, आप इन चरणों को छोड़ सकते हैं।

![Agent Setup](../../../translated_images/agent-setup.68a0c72f47bd1383584c52f14d694b54ea96c56c49660222409f83451b8220a8.hi.png)

3. नया मल्टी-AI एजेंट बनाने के लिए, बस **New Agent** पर क्लिक करें। नया बनाया गया एजेंट Agents पेज पर दिखेगा।

## एजेंट का परीक्षण करें

एजेंट बनाने के बाद, आप इसे Azure AI Foundry पोर्टल के प्लेग्राउंड में उपयोगकर्ता प्रश्नों पर प्रतिक्रिया देने के लिए परीक्षण कर सकते हैं।

1. अपने एजेंट के **Setup** पैन के शीर्ष पर, **Try in playground** चुनें।
2. **Playground** पैन में, आप चैट विंडो में प्रश्न टाइप करके एजेंट के साथ बातचीत कर सकते हैं। उदाहरण के लिए, आप एजेंट से 28 तारीख को सिएटल से न्यूयॉर्क के लिए फ्लाइट खोजने को कह सकते हैं।

    > **Note**: एजेंट सटीक उत्तर न दे, क्योंकि इस अभ्यास में कोई रियल-टाइम डेटा उपयोग नहीं किया जा रहा है। उद्देश्य एजेंट की उपयोगकर्ता प्रश्नों को समझने और दिए गए निर्देशों के आधार पर जवाब देने की क्षमता का परीक्षण करना है।

    ![Agent Playground](../../../translated_images/agent-playground.847acb21209744353080ead65ec9326b917a6b90121d4b63f6f412a4d65af2a0.hi.png)

3. एजेंट का परीक्षण करने के बाद, आप इसे और बेहतर बनाने के लिए अधिक intents, प्रशिक्षण डेटा, और actions जोड़ सकते हैं।

## संसाधनों को साफ़ करें

जब आपने एजेंट का परीक्षण पूरा कर लिया हो, तो अतिरिक्त लागत से बचने के लिए इसे हटा सकते हैं।
1. [Azure portal](https://portal.azure.com) खोलें और उस रिसोर्स ग्रुप की सामग्री देखें जहाँ आपने इस अभ्यास के लिए हब संसाधन तैनात किए थे।
2. टूलबार पर, **Delete resource group** चुनें।
3. रिसोर्स ग्रुप का नाम दर्ज करें और पुष्टि करें कि आप इसे हटाना चाहते हैं।

## संसाधन

- [Azure AI Foundry documentation](https://learn.microsoft.com/en-us/azure/ai-studio/?WT.mc_id=academic-105485-koreyst)
- [Azure AI Foundry portal](https://ai.azure.com/?WT.mc_id=academic-105485-koreyst)
- [Getting Started with Azure AI Studio](https://techcommunity.microsoft.com/blog/educatordeveloperblog/getting-started-with-azure-ai-studio/4095602?WT.mc_id=academic-105485-koreyst)
- [Fundamentals of AI agents on Azure](https://learn.microsoft.com/en-us/training/modules/ai-agent-fundamentals/?WT.mc_id=academic-105485-koreyst)
- [Azure AI Discord](https://aka.ms/AzureAI/Discord)

**अस्वीकरण**:  
इस दस्तावेज़ का अनुवाद AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके किया गया है। हम सटीकता के लिए प्रयासरत हैं, लेकिन कृपया ध्यान दें कि स्वचालित अनुवादों में त्रुटियाँ या अशुद्धियाँ हो सकती हैं। मूल दस्तावेज़ अपनी मूल भाषा में अधिकारिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम जिम्मेदार नहीं हैं।