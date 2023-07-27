import vertexai
from vertexai.preview.language_models import TextGenerationModel

def predict_large_language_model_sample(
    project_id: str,
    model_name: str,
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    location: str = "us-central1",
    tuned_model_name: str = "",
    ) :
    """Predict using a Large Language Model."""
    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained(model_name)
    if tuned_model_name:
      model = model.get_tuned_model(tuned_model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,)
    print(f"Response from Model: {response.text}")
predict_large_language_model_sample("cloud-llm-preview4", "text-bison@001", 0.2, 256, 0.95, 40, '''Provide a summary with about two sentences for the following article: Beyond our own products, we think it\'s important to make it easy, safe and scalable for others to benefit from these advances by building on top of our best models. Next month, we\'ll start onboarding individual developers, creators and enterprises so they can try our Generative Language API, initially powered by LaMDA with a range of models to follow. Over time, we intend to create a suite of tools and APIs that will make it easy for others to build more innovative applications with AI. Having the necessary compute power to build reliable and trustworthy AI systems is also crucial to startups, and we are excited to help scale these efforts through our Google Cloud partnerships with Cohere, C3.ai and Anthropic, which was just announced last week. Stay tuned for more developer details soon.
Summary: Google is making its AI technology more accessible to developers, creators, and enterprises. Next month, Google will start onboarding developers to try its Generative Language API, which will initially be powered by LaMDA. Over time, Google intends to create a suite of tools and APIs that will make it easy for others to build more innovative applications with AI. Google is also excited to help scale these efforts through its Google Cloud partnerships with Cohere, C3.ai, and Anthropic.

Provide a summary with about two sentences for the following article: The benefits of electricPromptData kitchens go beyond climate impact, starting with speed. The first time I ever cooked on induction (electric) equipment, the biggest surprise was just how incredibly fast it is. In fact, induction boils water twice as fast as traditional gas equipment and is far more efficient — because unlike a flame, electric heat has nowhere to escape. At Bay View, our training programs help Google chefs appreciate and adjust to the new pace of induction. The speed truly opens up whole new ways of cooking.
Summary: Electric kitchens are faster, more efficient, and better for the environment than gas kitchens. Induction cooking is particularly fast, boiling water twice as fast as traditional gas equipment. This speed opens up whole new ways of cooking. Google chefs are trained to appreciate and adjust to the new pace of induction cooking at Bay View.

Provide a summary with about two sentences for the following article: We\'re also using AI to forecast floods, another extreme weather pattern exacerbated by climate change. We\'ve already helped communities to predict when floods will hit and how deep the waters will get — in 2021, we sent 115 million flood alert notifications to 23 million people over Google Search and Maps, helping save countless lives. Today, we\'re sharing that we\'re now expanding our coverage to more countries in South America (Brazil and Colombia), Sub-Saharan Africa (Burkina Faso, Cameroon, Chad, Democratic Republic of Congo, Ivory Coast, Ghana, Guinea, Malawi, Nigeria, Sierra Leone, Angola, South Sudan, Namibia, Liberia, and South Africa), and South Asia (Sri Lanka). We\'ve used an AI technique called transfer learning to make it work in areas where there\'s less data available. We\'re also announcing the global launch of Google FloodHub, a new platform that displays when and where floods may occur. We\'ll also be bringing this information to Google Search and Maps in the future to help more people to reach safety in flooding situations.
Summary: Google is using AI to forecast floods in South America, Sub-Saharan Africa, South Asia, and other parts of the world. The AI technique of transfer learning is being used to make it work in areas where there\'s less data available. Google FloodHub, a new platform that displays when and where floods may occur, has also been launched globally. This information will also be brought to Google Search and Maps in the future to help more people reach safety in flooding situations.

Provide a summary with about two sentences for the following article: In order to learn skiing, you must first be educated on the proper use of the equipment. This includes learning how to properly fit your boot on your foot, understand the different functions of the ski, and bring gloves, goggles etc. Your instructor starts you with one-footed ski drills. Stepping side-to-side, forward-and-backward, making snow angels while keeping your ski flat to the ground, and gliding with the foot not attached to a ski up for several seconds. Then you can put on both skis and get used to doing them with two skis on at once. Next, before going down the hill, you must first learn how to walk on the flat ground and up small hills through two methods, known as side stepping and herringbone. Now it\'s time to get skiing! For your first attempted run, you will use the skills you just learned on walking up the hill, to go down a small five foot vertical straight run, in which you will naturally stop on the flat ground. This makes you learn the proper athletic stance to balance and get you used to going down the hill in a safe, controlled setting. What do you need next? To be able to stop yourself. Here, your coach will teach you how to turn your skis into a wedge, also commonly referred to as a pizza, by rotating legs inward and pushing out on the heels. Once learned, you practice a gliding wedge down a small hill where you gradually come to a stop on the flat ground thanks to your wedge. Finally, you learn the necessary skill of getting up after falling, which is much easier than it looks, but once learned, a piece of cake.
Summary: Skiing is a great way to enjoy the outdoors and get some exercise. It can be a little daunting at first, but with a little practice, you\'ll be skiing like a pro in no time.

Provide a summary with about two sentences for the following article: Yellowstone National Park is an American national park located in the western United States, largely in the northwest corner of Wyoming and extending into Montana and Idaho. It was established by the 42nd U.S. Congress with the Yellowstone National Park Protection Act and signed into law by President Ulysses S. Grant on March 1, 1872. Yellowstone was the first national park in the U.S. and is also widely held to be the first national park in the world.The park is known for its wildlife and its many geothermal features, especially the Old Faithful geyser, one of its most popular. While it represents many types of biomes, the subalpine forest is the most abundant. It is part of the South Central Rockies forests ecoregion.
Summary: Yellowstone National Park is the first national park in the United States and the world. It is located in the western United States, largely in the northwest corner of Wyoming and extending into Montana and Idaho. The park is known for its wildlife and its many geothermal features, especially the Old Faithful geyser.

Provide a summary with about two sentences for the following article: The efficient-market hypothesis (EMH) is a hypothesis in financial economics that states that asset prices reflect all available information. A direct implication is that it is impossible to "beat the market" consistently on a risk-adjusted basis since market prices should only react to new information. Because the EMH is formulated in terms of risk adjustment, it only makes testable predictions when coupled with a particular model of risk. As a result, research in financial economics since at least the 1990s has focused on market anomalies, that is, deviations from specific models of risk. The idea that financial market returns are difficult to predict goes back to Bachelier, Mandelbrot, and Samuelson, but is closely associated with Eugene Fama, in part due to his influential 1970 review of the theoretical and empirical research. The EMH provides the basic logic for modern risk-based theories of asset prices, and frameworks such as consumption-based asset pricing and intermediary asset pricing can be thought of as the combination of a model of risk with the EMH. Many decades of empirical research on return predictability has found mixed evidence. Research in the 1950s and 1960s often found a lack of predictability (e.g. Ball and Brown 1968; Fama, Fisher, Jensen, and Roll 1969), yet the 1980s-2000s saw an explosion of discovered return predictors (e.g. Rosenberg, Reid, and Lanstein 1985; Campbell and Shiller 1988; Jegadeesh and Titman 1993). Since the 2010s, studies have often found that return predictability has become more elusive, as predictability fails to work out-of-sample (Goyal and Welch 2008), or has been weakened by advances in trading technology and investor learning (Chordia, Subrahmanyam, and Tong 2014; McLean and Pontiff 2016; Martineau 2021).
Summary:
''', "us-central1")