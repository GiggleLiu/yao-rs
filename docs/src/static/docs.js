/* Progressive enhancement only; content and mdBook navigation work without it. */
(() => {
    const title = document.querySelector(".menu-title");
    if (title) {
        const brand = document.createElement("a");
        brand.className = "site-brand";
        brand.href = `${path_to_root}index.html`;
        brand.textContent = "yao-rs";
        const label = document.createElement("span");
        label.textContent = "Documentation";
        brand.append(label);
        title.replaceChildren(brand);
    }

    const darkTheme = document.getElementById("mdbook-theme-coal");
    if (darkTheme) darkTheme.textContent = "Dark";
    const search = document.getElementById("mdbook-searchbar");
    if (search) search.placeholder = "Search documentation…";

    // mdBook owns folding; expose the existing control to keyboard users.
    document.querySelectorAll(".chapter-fold-toggle").forEach((toggle) => {
        const item = toggle.closest(".chapter-item");
        const name = toggle.previousElementSibling.textContent;
        toggle.setAttribute("role", "button");
        toggle.setAttribute("tabindex", "0");
        toggle.setAttribute("aria-label", `Toggle ${name}`);
        const update = () => toggle.setAttribute("aria-expanded", String(item.classList.contains("expanded")));
        update();
        toggle.addEventListener("click", update);
        toggle.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                toggle.click();
            }
        });
    });

    const main = document.querySelector("main");
    if (!main) return;
    // Preserve the SVG's drawing scale. Long circuits scroll instead of shrinking
    // their gate labels, and short circuits keep their intrinsic dimensions.
    main.querySelectorAll('img[src*="/generated/svg/"]').forEach((img) => {
        if (img.closest(".circuit-preview")) return;
        const frame = document.createElement("span");
        frame.className = "circuit-diagram";
        frame.tabIndex = 0;
        frame.setAttribute("role", "region");
        frame.setAttribute("aria-label", `${img.alt}. Scroll horizontally to explore the circuit.`);
        frame.addEventListener("keydown", (event) => {
            // Let the browser scroll this diagram instead of mdBook changing chapters.
            if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
                event.stopPropagation();
            }
        });
        img.replaceWith(frame);
        frame.append(img);
    });
    const edit = document.querySelector('a[rel="edit"]');
    const footer = document.createElement("footer");
    footer.className = "page-footer";
    const note = document.createElement("span");
    note.textContent = "yao-rs · MIT license";
    footer.append(note);
    if (edit) {
        edit.textContent = "Edit this page";
        footer.append(edit);
    }
    main.append(footer);

    const headings = [...main.querySelectorAll("h2[id]")];
    if (headings.length < 3) return;
    const outline = document.createElement("nav");
    outline.className = "page-outline";
    outline.setAttribute("aria-label", "On this page");
    const heading = document.createElement("p");
    heading.textContent = "On this page";
    outline.append(heading);
    const links = headings.map((section) => {
        const link = document.createElement("a");
        link.href = `#${section.id}`;
        link.textContent = section.textContent;
        outline.append(link);
        return link;
    });
    main.parentElement.append(outline);
    if (!("IntersectionObserver" in window)) return;
    const observer = new IntersectionObserver((entries) => {
        for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            links.forEach((link) => link.removeAttribute("aria-current"));
            links[headings.indexOf(entry.target)].setAttribute("aria-current", "location");
        }
    }, { rootMargin: "-80px 0px -65% 0px" });
    headings.forEach((section) => observer.observe(section));
})();
