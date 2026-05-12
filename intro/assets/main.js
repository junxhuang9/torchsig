const STORAGE_KEY = "torchsig-intro-theme";
const applyTheme = (theme) => {
  document.body.classList.toggle("light", theme === "light");
};

const savedTheme = localStorage.getItem(STORAGE_KEY);
if (savedTheme) {
  applyTheme(savedTheme);
}

document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
  button.addEventListener("click", () => {
    const nextTheme = document.body.classList.contains("light") ? "dark" : "light";
    localStorage.setItem(STORAGE_KEY, nextTheme);
    applyTheme(nextTheme);
  });
});
