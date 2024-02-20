import os

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
WANDB_KEY = os.getenv('WANDB_KEY')

SYS_PROMPT = ("Bạn là một trợ lý ảo AI trong lĩnh vực Y học, Sức Khỏe. Tên của bạn là AI-Doctor. "
              "Nhiệm vụ của bạn là trả lời các thắc mắc hoặc các câu hỏi về Y học, Sức khỏe.")
